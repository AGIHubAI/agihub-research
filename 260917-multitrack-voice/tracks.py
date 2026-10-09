"""
Multi-track local voice pipeline: three mlx 4-bit models in parallel tracks.

- Filler track, gemma-3-270m: picks a short phrase from a bank within ~100 ms.
  A backchannel phrase ("I see.", "Yes.") ends the turn. A hold phrase
  ("Let me think.") covers the gap while the speaker answers. Bank audio is
  pre-synthesized, so the filler costs no TTS time.
- Reasoner track, qwen3.5-4b: starts with the turn, has tools (time, calculator,
  document search) and writes its answer to the shared context. Its first token
  (`<tool_call>` or not, ~150 ms with a cached tool prompt) is the routing signal.
- Speaker track, gemma-3-1b: writes the spoken answer, streamed sentence by
  sentence into Kokoro. When the reasoner calls a tool, the speaker waits for
  the reasoner's answer and voices it. It also reads earlier turns' notes.

Neither small model can route on its own: the 1b never said CHECKING for
tool questions (0/9, it guessed wrong), and with few-shot examples it deferred
6 of 8 plain facts.

All models live in one process and generate from separate threads (measured
safe on mlx 0.31.3). Imported by notebook.ipynb and bench_independent.py.
"""
from __future__ import annotations

import ast
import difflib
import json
import operator
import re
import sys
import threading
import time
import queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # self-contained: helpers live next to this file

# Sets the espeak env vars before kokoro imports.
from qwen_pipeline import ASR, ms, text_hash, load_items as load_fact_items  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from segment import Segmenter  # noqa: E402
from tts import synthesize, trim_padding  # noqa: E402

AUDIO_DIR = HERE / "audio"
TTS_SAMPLE_RATE = 24000
KOKORO_VOICE = "af_heart"

FILLER_REPO = "mlx-community/gemma-3-270m-it-qat-4bit"
SPEAKER_REPO = "mlx-community/gemma-3-1b-it-qat-4bit"
REASONER_REPO = "mlx-community/Qwen3.5-4B-4bit"

DOCS = [HERE / "reference" / "qwen35-WHITEPAPER.md",
        HERE / "reference" / "qwen35-README.md"]

# --------------------------------------------------------------------------
# Items: 8 facts (hold), 4 remarks (backchannel), 3 tool questions (hold)
# --------------------------------------------------------------------------

EXTRA_ITEMS = [
    {"id": "back-001", "text": "I just got back from a long trip.", "route": "backchannel"},
    {"id": "back-002", "text": "Thanks, that was really helpful.", "route": "backchannel"},
    {"id": "back-003", "text": "Can you hear me?", "route": "backchannel"},
    {"id": "back-004", "text": "Okay, got it.", "route": "backchannel"},
    {"id": "tool-001", "text": "What is seventeen times twenty three?", "route": "hold",
     "expect": {"contains_any": ["391", "three hundred ninety-one",
                                 "three hundred and ninety-one"]}, "tool": "calculate"},
    {"id": "tool-002", "text": "What time is it right now?", "route": "hold",
     "expect": {"time": True}, "tool": "get_time"},
    {"id": "tool-003", "text": "In the qwen spike, what was the time to first audio on mlx?",
     "route": "hold", "expect": {"contains_any": ["860"]}, "tool": "search_docs"},
]


def load_items() -> list[dict]:
    facts = [dict(i, route="hold") for i in load_fact_items()]
    return facts + EXTRA_ITEMS


def is_correct(item: dict, reply: str) -> bool | None:
    expect = item.get("expect")
    if not expect:
        return None  # backchannel: judged by route only
    if expect.get("time"):
        now = datetime.now()
        h12 = str(now.hour % 12 or 12)
        return bool(re.search(rf"\b({now.hour}|{h12})\b", reply))
    return any(w.lower() in reply.lower() for w in expect["contains_any"])


# --------------------------------------------------------------------------
# mlx model with an append-only prompt cache (qwen3.5 experiment finding)
# --------------------------------------------------------------------------

class MlxModel:
    def __init__(self, repo: str, *, template_kwargs: dict | None = None):
        self.repo = repo
        self.template_kwargs = template_kwargs or {}
        self.last: dict = {}
        self.lock = threading.Lock()  # one generation per model at a time

    def load(self) -> float:
        t = time.monotonic()
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.model, self.tok = load(self.repo)
        self.detok = self.tok.detokenizer  # cached once, see cycle 7
        self._make_cache = make_prompt_cache
        # gemma's tokenizer lists only <eos>; turns actually end on <end_of_turn>.
        self.stop_ids = set(self.tok.eos_token_ids)
        for marker in ("<end_of_turn>", "<|im_end|>"):
            tid = self.tok.convert_tokens_to_ids(marker)
            if isinstance(tid, int) and tid != self.tok.unk_token_id:
                self.stop_ids.add(tid)
                self.end_marker = marker
        self.reset()
        return ms(t)

    def reset(self):
        self.cache = self._make_cache(self.model)
        self.cached: list[int] = []
        self._last_messages: list[dict] = []
        self._last_reply = ""

    def render(self, messages: list[dict], **kwargs) -> str:
        return self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **self.template_kwargs, **kwargs,
        )

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _delta(self, messages: list[dict]) -> list[int] | None:
        if (not self.cached or len(messages) < 3 or messages[-2]["role"] != "assistant"
                or messages[-2].get("content") != self._last_reply
                or messages[:-2] != self._last_messages):
            return None
        rendered = self.render(messages)
        marker = self._last_reply + self.end_marker
        at = rendered.rfind(marker)
        if at < 0:
            return None
        return self.encode(rendered[at + len(marker):])

    def generate(self, tokens: list[int], *, max_tokens: int, cancel: threading.Event | None,
                 on_text=None, on_first_token=None) -> str:
        """Feed tokens into the current cache and stream text to on_text."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        started = time.monotonic()
        self.detok.reset()
        fed = list(tokens)
        out = 0
        ttft = None
        cancelled = False
        for token, _ in generate_step(mx.array(tokens), self.model,
                                      max_tokens=max_tokens, prompt_cache=self.cache):
            token = int(token)
            fed.append(token)  # already scheduled into the cache before the yield
            if ttft is None:
                ttft = ms(started)
                if on_first_token:
                    on_first_token(token)
            if token in self.stop_ids:
                break
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            out += 1
            self.detok.add_token(token)
            # last_segment consumes what it returns: read it exactly once per step
            segment = self.detok.last_segment
            if on_text and segment:
                on_text(segment)
        self.detok.finalize()
        segment = self.detok.last_segment
        if on_text and segment:
            on_text(segment)
        total = ms(started)
        self.cached += fed
        decode_s = (total - (ttft or 0)) / 1000
        self.last.update(ttft_ms=ttft, total_ms=total, out_tokens=out,
                         prefilled_tokens=len(tokens), cancelled=cancelled,
                         decode_tok_s=round(out / decode_s, 1) if decode_s > 0 and out else None)
        return self.detok.text

    def chat(self, messages: list[dict], *, max_tokens: int = 120,
             cancel: threading.Event | None = None, on_text=None, stateless=False) -> str:
        """Append-only cached chat turn. stateless=True prefills from scratch."""
        with self.lock:
            delta = None if stateless else self._delta(messages)
            hit = len(self.cached) if delta is not None else 0
            if delta is None:
                self.reset()
                delta = self.encode(self.render(messages))
            self.last = {"cache_hit_tokens": hit}
            text = self.generate(delta, max_tokens=max_tokens, cancel=cancel, on_text=on_text)
            if self.last["cancelled"] or stateless:
                self.reset()  # partial reply is in the cache; start clean next turn
            else:
                self._last_messages = list(messages)
                self._last_reply = text.strip()
            return text


# --------------------------------------------------------------------------
# TTS: one Kokoro thread, pre-synthesized phrase bank, per-chunk ready times
# --------------------------------------------------------------------------

BANK = {
    "hold": ["Let me think.", "Good question.", "One moment.", "Let me check that."],
    "backchannel": ["Yes.", "No.", "I see.", "Of course.", "Mm-hmm.", "Okay.",
                    "You're welcome."],
}
ROUTE_OF = {p: r for r, phrases in BANK.items() for p in phrases}
BANK_SPEED = 1.15
BANK_KEEP_MS = 40


@dataclass
class Voice:
    """Audio produced for one turn: (ready_ms, kind, text, samples)."""
    started: float
    chunks: list = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)

    def add(self, kind: str, text: str, audio: np.ndarray):
        self.chunks.append((ms(self.started), kind, text, audio))


class TTS:
    _END = object()

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self.bank: dict[str, np.ndarray] = {}
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    def _run(self):
        t = time.monotonic()
        from kokoro import KPipeline

        self._kokoro = KPipeline(lang_code="a", device="mps")
        self.load_ms = ms(t)
        self._ready.set()
        while True:
            voice, kind, text = self._q.get()
            if text is self._END:
                voice.done.set()
                continue
            if kind == "bank":
                # Filler clips: faster speech and 40 ms of edge padding instead of 200 ms.
                audio = np.concatenate([
                    np.asarray(r.audio if hasattr(r, "audio") else r[2], dtype=np.float32).reshape(-1)
                    for r in self._kokoro(text, voice=KOKORO_VOICE, speed=BANK_SPEED)])
                voice.add(kind, text, trim_padding(audio, keep_ms=BANK_KEEP_MS))
            else:
                voice.add(kind, text, trim_padding(synthesize(self._kokoro, text, KOKORO_VOICE)))

    def say(self, voice: Voice, kind: str, text: str):
        self._q.put((voice, kind, text))

    def end(self, voice: Voice):
        self._q.put((voice, None, self._END))

    def synth(self, text: str, kind: str = "sync") -> np.ndarray:
        voice = Voice(started=time.monotonic())
        self.say(voice, kind, text)
        self.end(voice)
        voice.done.wait()
        return voice.chunks[0][3]

    def prewarm_and_bank(self) -> dict:
        """Two warm passes, then synthesize every bank phrase once and keep it."""
        t = time.monotonic()
        self.synth("Warming up the voice.")
        self.synth("There are seven continents.")
        warm = ms(t)
        live = {}
        for phrase in ROUTE_OF:
            t = time.monotonic()
            self.bank[phrase] = self.synth(phrase, kind="bank")
            live[phrase] = ms(t)
        return {"kokoro_warm_ms": warm, "bank_phrases": len(self.bank),
                "bank_speed": BANK_SPEED, "bank_keep_ms": BANK_KEEP_MS,
                "bank_clip_ms": {p: round(len(a) / TTS_SAMPLE_RATE * 1000) for p, a in self.bank.items()},
                "bank_live_synth_ms": live,
                "bank_live_synth_median_ms": median(live.values())}


def input_audio(tts: TTS, item: dict) -> np.ndarray:
    AUDIO_DIR.mkdir(exist_ok=True)
    path = AUDIO_DIR / f"{item['id']}.wav"
    if path.exists():
        audio, _ = sf.read(path, dtype="float32")
    else:
        audio = tts.synth(item["text"])
        sf.write(path, audio, TTS_SAMPLE_RATE)
    return resample_poly(audio, 2, 3).astype(np.float32)


# --------------------------------------------------------------------------
# Filler track: gemma-3-270m, few-shot prefix pinned in a trimmable cache
# --------------------------------------------------------------------------

FILLER_SYSTEM = (
    "Reply with a very short filler phrase only. For a question use one of: Let me think. / "
    "Good question. / One moment. / Let me check that. For a remark use one of: Yes. / "
    "I see. / Of course. / Okay. / You're welcome."
)
# Chat-format few-shot. A single-message list prompt made the 270m answer "Okay." to 15/15
# items; this format routed 14/15 (it still often answers instead of picking a phrase,
# which the matcher routes to hold).
FILLER_SHOTS = [
    ("What is the tallest mountain in the world?", "Good question."),
    ("I had a really busy day at work.", "I see."),
    ("Thank you so much.", "You're welcome."),
    ("Are you still there?", "Yes."),
    ("What is twelve times nine?", "Let me check that."),
    ("Sounds good.", "Okay."),
    ("Who painted the Mona Lisa?", "Let me think."),
    ("How far away is the moon?", "One moment."),
]


def filler_messages(utterance: str) -> list[dict]:
    messages = [{"role": "system", "content": FILLER_SYSTEM}]
    for q, a in FILLER_SHOTS:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return messages + [{"role": "user", "content": utterance}]


class FillerTrack:
    def __init__(self):
        self.lm = MlxModel(FILLER_REPO)

    def load(self) -> float:
        t = time.monotonic()
        self.lm.load()
        prefix_text, self._suffix = self.lm.render(filler_messages("{utterance}")).split(
            "{utterance}")
        self._prefix = self.lm.encode(prefix_text)
        self._pin_prefix()
        return ms(t)

    def _pin_prefix(self):
        """Prefill the few-shot prefix once; every call trims back to it."""
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache

        self.lm.reset()
        self.trimmable = can_trim_prompt_cache(self.lm.cache)
        if not self.trimmable:
            return
        self.lm.model(mx.array(self._prefix)[None], cache=self.lm.cache)
        mx.eval([c.state for c in self.lm.cache])
        self.lm.cached = list(self._prefix)

    def pick(self, utterance: str) -> dict:
        from mlx_lm.models.cache import trim_prompt_cache

        started = time.monotonic()
        with self.lm.lock:
            if self.trimmable:
                tokens = self.lm.encode(utterance + self._suffix)
            else:
                self.lm.reset()
                tokens = self._prefix + self.lm.encode(utterance + self._suffix)
            raw = self.lm.generate(tokens, max_tokens=8, cancel=None).strip()
            if self.trimmable:
                trim_prompt_cache(self.lm.cache, len(self.lm.cached) - len(self._prefix))
                self.lm.cached = list(self._prefix)
        phrase, matched = match_phrase(raw)
        return {"filler_raw": raw, "filler_phrase": phrase, "filler_matched": matched,
                "route": ROUTE_OF[phrase], "filler_decision_ms": ms(started),
                "filler_ttft_ms": self.lm.last.get("ttft_ms"),
                "filler_prefilled_tokens": len(tokens)}


def match_phrase(raw: str) -> tuple[str, str]:
    first = raw.splitlines()[0].strip() if raw else ""
    norm = lambda s: re.sub(r"[^a-z' ]", "", s.lower()).strip()  # noqa: E731
    by_norm = {norm(p): p for p in ROUTE_OF}
    if norm(first) in by_norm:
        return by_norm[norm(first)], "exact"
    close = difflib.get_close_matches(norm(first), by_norm, n=1, cutoff=0.75)
    if close:
        return by_norm[close[0]], "fuzzy"
    # Anything else (usually the 270m answering the question itself) routes to hold.
    return "One moment.", "fallback"


# --------------------------------------------------------------------------
# Shared context + reasoner track: qwen3.5-4b with tools
# --------------------------------------------------------------------------

class SharedContext:
    def __init__(self):
        self._lock = threading.Lock()
        self.notes: list[dict] = []
        self._events: dict[str, threading.Event] = {}

    def event(self, turn: str) -> threading.Event:
        with self._lock:
            return self._events.setdefault(turn, threading.Event())

    def write(self, note: dict):
        with self._lock:
            self.notes.append(note)
        self.event(note["turn"]).set()

    def note(self, turn: str) -> dict | None:
        with self._lock:
            return next((n for n in reversed(self.notes) if n["turn"] == turn), None)

    def since(self, count: int) -> list[dict]:
        with self._lock:
            return self.notes[count:]


def _calc(expression: str) -> str:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return str(ev(ast.parse(expression.replace("x", "*"), mode="eval").body))


class DocIndex:
    """Keyword index over the qwen spike docs. No embeddings: Ollama stays unloaded."""

    def __init__(self, paths=DOCS):
        self.chunks = []
        for path in paths:
            for para in re.split(r"\n\s*\n", path.read_text()):
                if para.strip():
                    self.chunks.append((path.name, para.strip()))

    def search(self, query: str, k: int = 3) -> str:
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = sorted(
            ((sum(c.lower().count(w) for w in words), name, c) for name, c in self.chunks),
            reverse=True,
        )
        return "\n---\n".join(f"[{name}] {c[:600]}" for s, name, c in scored[:k] if s)


TOOLS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "Current local time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate an arithmetic expression, e.g. 17*23.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search this project's spike reports and whitepapers.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]

TOOL_CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
TOOL_PARAM = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.S)

REASONER_SYSTEM = ("You research answers for a voice assistant. Use tools for the current "
                   "time, arithmetic, and questions about this project. Reply with the "
                   "answer in one short plain sentence.")


TOOL_CALL_TOKEN = "<tool_call>"


class ReasonerTrack:
    def __init__(self, context: SharedContext, thinking: bool = False):
        self.lm = MlxModel(REASONER_REPO, template_kwargs={"enable_thinking": thinking})
        self.context = context
        self.docs = DocIndex()
        self.thinking = thinking

    def load(self) -> float:
        t = time.monotonic()
        self.lm.load()
        self._pin_prefix()
        return ms(t)

    def _pin_prefix(self):
        """Prefill system prompt + tool schema (~400 tokens) once and keep a snapshot.

        The hybrid cache cannot be trimmed, but its state arrays can be copied:
        each call starts from a copy (~0.3 ms) instead of a 1.2 s re-prefill.
        """
        import mlx.core as mx

        rendered = self.lm.render([{"role": "system", "content": REASONER_SYSTEM},
                                   {"role": "user", "content": "{q}"}], tools=TOOLS)
        self._prefix_text, self._suffix = rendered.split("{q}")
        self._prefix = self.lm.encode(self._prefix_text)
        self._snapshot = self.lm._make_cache(self.lm.model)
        self.lm.model(mx.array(self._prefix)[None], cache=self._snapshot)
        mx.eval([c.state for c in self._snapshot])
        self.tool_call_id = self.lm.tok.convert_tokens_to_ids(TOOL_CALL_TOKEN)

    def _clone(self):
        import mlx.core as mx

        cache = self.lm._make_cache(self.lm.model)
        for dst, src in zip(cache, self._snapshot):
            state = src.state
            dst.state = ([None if a is None else mx.array(a) for a in state]
                         if isinstance(state, list) else tuple(mx.array(a) for a in state))
            if hasattr(src, "offset"):
                dst.offset = src.offset
        return cache

    def _run_tool(self, name: str, args: dict) -> str:
        if name == "get_time":
            return datetime.now().strftime("%H:%M")
        if name == "calculate":
            return _calc(args.get("expression", ""))
        if name == "search_docs":
            return self.docs.search(args.get("query", "")) or "no results"
        return f"unknown tool {name}"

    def answer(self, question: str, *, on_route=None, cancel: threading.Event | None = None,
               max_rounds: int = 3) -> dict:
        started = time.monotonic()
        messages = [{"role": "system", "content": REASONER_SYSTEM},
                    {"role": "user", "content": question}]
        tools_used, text, route = [], "", {}

        def first(token: int):
            route.update(tool=token == self.tool_call_id, route_ms=ms(started))
            if on_route:
                on_route(route["tool"])

        with self.lm.lock:
            # Round 1 from the pinned prefix; tool rounds re-render (they are rare
            # and the tool result itself sits between prefix and continuation).
            self.lm.cache = self._clone()
            text = self.lm.generate(self.lm.encode(question + self._suffix), max_tokens=256,
                                    cancel=cancel, on_first_token=first)
            ttft = self.lm.last.get("ttft_ms")
            for _ in range(max_rounds):
                calls = TOOL_CALL.findall(text)
                if not calls or (cancel is not None and cancel.is_set()):
                    break
                tool_calls, results = [], []
                for name, body in calls:
                    args = {k: v.strip() for k, v in TOOL_PARAM.findall(body)}
                    tool_calls.append({"type": "function",
                                       "function": {"name": name, "arguments": args}})
                    try:
                        result = self._run_tool(name, args)
                    except Exception as error:  # a bad expression must not kill the track
                        result = f"error: {error}"
                    results.append(result)
                    tools_used.append({"name": name, "args": args, "result": result[:200]})
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                messages += [{"role": "tool", "content": r} for r in results]
                self.lm.cache = self._clone()
                rendered = self.lm.render(messages, tools=TOOLS)
                assert rendered.startswith(self._prefix_text), "tool prompt prefix changed"
                text = self.lm.generate(self.lm.encode(rendered[len(self._prefix_text):]),
                                        max_tokens=256, cancel=cancel)
            self.lm.reset()
        answer = re.sub(r"<think>.*?</think>", "", TOOL_CALL.sub("", text), flags=re.S).strip()
        return {"answer": answer, "tools": tools_used, "reasoner_ttft_ms": ttft,
                "reasoner_route_ms": route.get("route_ms"), "reasoner_tool": route.get("tool"),
                "reasoner_total_ms": ms(started),
                "reasoner_cancelled": bool(cancel is not None and cancel.is_set())}

    def run(self, turn: str, question: str, started: float, decided: threading.Event,
            decision: dict, cancel: threading.Event):
        def on_route(tool: bool):
            decision.update(tool=tool, ready_ms=ms(started))
            decided.set()

        result = self.answer(question, on_route=on_route, cancel=cancel)
        decided.set()  # also on failure before the first token
        result.update(turn=turn, question=question, ready_ms=ms(started))
        self.context.write(result)


# --------------------------------------------------------------------------
# Speaker track: gemma-3-1b
# --------------------------------------------------------------------------

SPEAKER_SYSTEM = "You are the voice of a voice assistant. Answer in one short sentence of plain words."


class SpeakerTrack:
    def __init__(self, context: SharedContext):
        self.lm = MlxModel(SPEAKER_REPO)
        self.context = context
        self.history: list[dict] = []
        self._seen_notes = 0

    def load(self) -> float:
        t = time.monotonic()
        self.lm.load()
        self.lm.chat([{"role": "system", "content": SPEAKER_SYSTEM},
                      {"role": "user", "content": "Say ready."}])
        self.lm.reset()
        return ms(t)

    def new_conversation(self):
        self.history = []
        self._seen_notes = len(self.context.notes)
        self.lm.reset()

    def _messages(self) -> list[dict]:
        return [{"role": "system", "content": SPEAKER_SYSTEM}] + self.history

    def speak(self, user_text: str, tts: TTS, voice: Voice, kind: str, *,
              result: str | None = None, exclude_turn: str | None = None) -> tuple[str, dict]:
        """One streamed reply into TTS. `result` is a reasoner answer to voice."""
        notes = [n for n in self.context.since(self._seen_notes)
                 if n["turn"] != exclude_turn and n["answer"] and not n["reasoner_cancelled"]]
        self._seen_notes = len(self.context.notes)
        prefix = ""
        if notes:
            prefix += "(Earlier research: " + " ".join(
                f"{n['question']} -> {n['answer']}" for n in notes[-2:]) + ")\n"
        if result is not None:
            prefix += f"(Research result for this question: {result})\n"
        self.history.append({"role": "user", "content": prefix + user_text})

        seg = Segmenter()
        first = {"ms": None}

        def emit(sentences):
            for sentence in sentences:
                first["ms"] = first["ms"] or ms(voice.started)
                tts.say(voice, kind, sentence)

        text = self.lm.chat(self._messages(), on_text=lambda f: emit(seg.feed(f))).strip()
        emit(seg.flush())
        self.history.append({"role": "assistant", "content": text})
        return text, dict(self.lm.last, first_sentence_ms=first["ms"])


# --------------------------------------------------------------------------
# One turn across all tracks
# --------------------------------------------------------------------------

def play_timeline(chunks: list) -> list[dict]:
    """Serial playback: each chunk starts when ready and when the previous one ends."""
    t, out = 0.0, []
    for ready, kind, text, audio in sorted(chunks, key=lambda c: c[0]):
        start = max(ready, t)
        dur = len(audio) / TTS_SAMPLE_RATE * 1000
        out.append({"kind": kind, "text": text, "ready_ms": ready, "start_ms": round(start, 1),
                    "dur_ms": round(dur, 1)})
        t = start + dur
    return out


POLICIES = {
    # A: every turn gets a filler as soon as the 270m decides; the reasoner always finishes.
    "always_filler": {"hold_filler": "always", "cancel_reasoner_without_tool": False},
    # B: backchannels play at once; a hold filler plays only when the reasoner calls a
    #    tool (long wait ahead); without a tool the reasoner is cancelled so the speaker
    #    has the GPU.
    "route_aware": {"hold_filler": "tool_only", "cancel_reasoner_without_tool": True},
}


class MultiTrack:
    def __init__(self, asr: ASR, tts: TTS, filler: FillerTrack, speaker: SpeakerTrack,
                 reasoner: ReasonerTrack, context: SharedContext, policy: str = "route_aware"):
        self.asr, self.tts, self.filler = asr, tts, filler
        self.speaker, self.reasoner, self.context = speaker, reasoner, context
        self.policy_name, self.policy = policy, POLICIES[policy]

    def turn(self, item: dict, audio_16k: np.ndarray, *, round_: int) -> dict:
        t0 = time.monotonic()
        turn_id = f"r{round_}-{item['id']}"
        transcript = self.asr.transcribe(audio_16k)
        rec: dict = {"id": item["id"], "round": round_, "expected_route": item["route"],
                     "transcript": transcript, "stt_ms": ms(t0)}
        voice = Voice(started=t0)

        # Reasoner starts immediately: its first token is the tool-routing signal.
        decided, decision, cancel = threading.Event(), {}, threading.Event()
        reasoner = threading.Thread(target=self.reasoner.run, daemon=True,
                                    args=(turn_id, transcript, t0, decided, decision, cancel))
        reasoner.start()

        pick = self.filler.pick(transcript)
        pick["filler_ready_ms"] = ms(t0)
        rec.update(pick, policy=self.policy_name)
        clip = self.tts.bank[pick["filler_phrase"]]
        if pick["route"] == "backchannel" or self.policy["hold_filler"] == "always":
            voice.add("filler", pick["filler_phrase"], clip)

        reply = ""
        if pick["route"] == "backchannel":
            cancel.set()
        else:
            decided.wait(timeout=5)
            rec.update(tool_route=decision.get("tool"), tool_route_ms=decision.get("ready_ms"))
            if decision.get("tool") and self.policy["hold_filler"] == "tool_only":
                voice.add("filler", pick["filler_phrase"], clip)
            if not decision.get("tool") and self.policy["cancel_reasoner_without_tool"]:
                cancel.set()
            if decision.get("tool"):
                self.context.event(turn_id).wait(timeout=60)
                note = self.context.note(turn_id)
                rec["deferred_result_ms"] = ms(t0)
                reply, stats = self.speaker.speak(
                    transcript, self.tts, voice, "deferred",
                    result=note["answer"] if note else "unknown", exclude_turn=turn_id)
            else:
                reply, stats = self.speaker.speak(transcript, self.tts, voice, "answer",
                                                  exclude_turn=turn_id)
            rec.update({f"speaker_{k}": v for k, v in stats.items()})
        self.tts.end(voice)
        voice.done.wait()
        reasoner.join(timeout=60)
        note = self.context.note(turn_id)
        if note:
            rec.update(reasoner_answer=note["answer"], reasoner_ready_ms=note["ready_ms"],
                       reasoner_ttft_ms=note["reasoner_ttft_ms"],
                       reasoner_total_ms=note["reasoner_total_ms"],
                       reasoner_cancelled=note["reasoner_cancelled"],
                       reasoner_tools=[t["name"] for t in note["tools"]])

        timeline = play_timeline(voice.chunks)
        spoken = [c for c in timeline if c["kind"] in ("answer", "deferred")]
        rec.update(
            timeline=timeline,
            first_audio_ms=timeline[0]["start_ms"] if timeline else None,
            answer_ready_ms=min((c["ready_ms"] for c in spoken), default=None),
            answer_start_ms=min((c["start_ms"] for c in spoken), default=None),
            end_to_end_ms=ms(t0),
            reply_text=reply,
            reply_hash=text_hash(reply),
            route_correct=rec["route"] == item["route"],
            # a hold item misrouted to backchannel has no answer, so it counts as wrong
            answer_correct=is_correct(item, reply) if item["route"] == "hold" else None,
        )
        AUDIO_DIR.mkdir(exist_ok=True)
        wav = AUDIO_DIR / f"combined-{turn_id}__reply.wav"
        sf.write(wav, np.concatenate([c[3] for c in sorted(voice.chunks, key=lambda c: c[0])]),
                 TTS_SAMPLE_RATE)
        rec["reply_wav"] = str(wav.relative_to(HERE))
        return rec


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(median(vals), 1) if vals else None


def summarize(turns: list[dict]) -> dict:
    facts = [t for t in turns if t["id"].startswith("fact-")]
    tools = [t for t in turns if t["id"].startswith("tool-")]
    back = [t for t in turns if t["expected_route"] == "backchannel"]
    direct = [t for t in facts if t["route"] == "hold" and not t.get("tool_route")]
    tool_of = {i["id"]: i["tool"] for i in EXTRA_ITEMS if "tool" in i}
    return {
        "turns": len(turns),
        "route_accuracy": f"{sum(t['route_correct'] for t in turns)}/{len(turns)}",
        "filler_fallback": sum(t["filler_matched"] == "fallback" for t in turns),
        "stt_ms": _med(t["stt_ms"] for t in turns),
        "filler_decision_ms": _med(t["filler_decision_ms"] for t in turns),
        "first_audio_ms": _med(t["first_audio_ms"] for t in turns),
        "first_audio_ms_backchannel": _med(t["first_audio_ms"] for t in back),
        "tool_route_ms": _med(t.get("tool_route_ms") for t in turns),
        "facts_tool_routed": f"{sum(bool(t.get('tool_route')) for t in facts)}/{len(facts)}",
        "facts_answer_ready_ms": _med(t["answer_ready_ms"] for t in direct),
        "facts_answer_start_ms": _med(t["answer_start_ms"] for t in direct),
        "facts_speaker_ttft_ms": _med(t.get("speaker_ttft_ms") for t in direct),
        "facts_correct": f"{sum(bool(t['answer_correct']) for t in facts)}/{len(facts)}",
        "tools_tool_routed": f"{sum(bool(t.get('tool_route')) for t in tools)}/{len(tools)}",
        "tools_expected_tool_used": f"{sum(tool_of[t['id']] in t.get('reasoner_tools', []) for t in tools)}/{len(tools)}",
        "tools_answer_start_ms": _med(t["answer_start_ms"] for t in tools),
        "tools_correct": f"{sum(bool(t['answer_correct']) for t in tools)}/{len(tools)}",
        "reasoner_ttft_ms": _med(t.get("reasoner_ttft_ms") for t in turns
                                 if not t.get("reasoner_cancelled")),
        "reasoner_ready_ms_tools": _med(t.get("reasoner_ready_ms") for t in tools),
    }
