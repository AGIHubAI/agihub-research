"""
ASR -> LLM -> TTS voice pipeline on qwen3.5:4b, every known latency
optimization applied, one monotonic clock per turn.

Imported by notebook.ipynb. Two LLM paths behind one streaming interface:

- `OllamaLLM`: qwen3.5:4b (Q4_K_M) over Ollama HTTP.
- `MlxLLM`: mlx-community/Qwen3.5-4B-4bit in-process via mlx-lm, one cached
  detokenizer and a persistent prompt cache (cycle 7 serving-path-ab).

The clock starts at "end of speech": the input WAV is already in memory,
so endpointing/VAD delay is not measured here (it is a fixed 1.5 s in the
live agents and is reported as a constant, not added).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# Kokoro's bundled espeakng-loader points at a CI path that does not exist
# here. Must be set before anything imports phonemizer.
os.environ.setdefault(
    "PHONEMIZER_ESPEAK_LIBRARY",
    "/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib",
)
os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", "/opt/homebrew/share/espeak-ng-data")
os.environ.setdefault("ESPEAK_DATA_PATH", "/opt/homebrew/share/espeak-ng-data")

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # self-contained: helpers live next to this file

from segment import Segmenter  # noqa: E402
from tts import synthesize, trim_padding  # noqa: E402

AUDIO_DIR = HERE / "audio"
RESULTS_PATH = HERE / "results.json"
DATASET = HERE / "data" / "dataset.jsonl"

TTS_SAMPLE_RATE = 24000
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"

# Optional accent LoRA weights (not published). Unset -> public whisper-small.en.
LORA_WHISPER = Path(os.environ.get("ACCENT_LORA_DIR", "/nonexistent"))
PUBLIC_WHISPER = "mlx-community/whisper-small.en-mlx"

OLLAMA_MODEL = "qwen3.5:4b"
MLX_MODEL = "mlx-community/Qwen3.5-4B-4bit"

MAX_TOKENS = 200
NUM_CTX = 4096
ENDPOINTING_MS = 1500  # live agents' fixed endpointing delay, not measured here

# Short spoken replies: the detail belongs elsewhere, not in the audio.
SYSTEM_PROMPT = (
    "You are a voice assistant. Answer in one short sentence. "
    "Plain words only: no lists, no markdown, no emoji."
)

OPTIMIZATIONS = {
    "asr_mlx_whisper_metal": True,
    "asr_warmup_transcribe": True,
    "llm_thinking_off": True,
    "llm_streaming_first_content_token": True,
    "llm_append_only_history": True,
    "llm_token_cap": MAX_TOKENS,
    "ollama_keep_alive_forever": True,
    "ollama_fixed_num_ctx": NUM_CTX,
    "mlx_in_process": True,
    "mlx_cached_detokenizer": True,
    "mlx_persistent_prompt_cache": True,
    "tts_kokoro_mps": True,
    "tts_sentence_streaming_worker_thread": True,
    "tts_trim_padding": True,
    "prewarm_all_stages": True,
}


def ms(since: float) -> float:
    return round((time.monotonic() - since) * 1000, 1)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_items() -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["id"].startswith("fact-")]


def is_correct(reply: str, item: dict) -> bool:
    return any(w.lower() in reply.lower() for w in item["expect"]["contains_any"])


def whisper_repo() -> tuple[str, str]:
    if (LORA_WHISPER / "weights.safetensors").exists():
        return str(LORA_WHISPER), "accent-lora"
    return PUBLIC_WHISPER, "public-baseline"


# --------------------------------------------------------------------------
# TTS: one thread owns Kokoro for the whole session
# --------------------------------------------------------------------------

@dataclass
class TurnAudio:
    started: float
    first_audio_ms: float | None = None
    synth_ms: float = 0.0
    chunks: list = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)


class TTSWorker:
    _STOP = object()
    _END = object()

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self.load_ms: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        started = time.monotonic()
        from kokoro import KPipeline

        self._kokoro = KPipeline(lang_code=KOKORO_LANG, device="mps")
        self.load_ms = ms(started)
        self._ready.set()

        while True:
            item = self._q.get()
            if item is self._STOP:
                return
            turn, text = item
            if text is self._END:
                turn.done.set()
                continue
            t = time.monotonic()
            audio = trim_padding(synthesize(self._kokoro, text, KOKORO_VOICE))
            turn.synth_ms += ms(t)
            if turn.first_audio_ms is None:
                turn.first_audio_ms = ms(turn.started)
            turn.chunks.append(audio)

    def say(self, turn: TurnAudio, sentence: str):
        self._q.put((turn, sentence))

    def end(self, turn: TurnAudio):
        self._q.put((turn, self._END))

    def prewarm(self) -> float:
        """Two passes: the first MPS synth is ~930 ms, the second ~165 ms."""
        t = time.monotonic()
        self.synth_blocking("Warming up the voice.")
        self.synth_blocking("There are seven continents.")
        return ms(t)

    def synth_blocking(self, text: str) -> np.ndarray:
        """Used for input audio generation and prewarm; same thread as turns."""
        turn = TurnAudio(started=time.monotonic())
        self.say(turn, text)
        self.end(turn)
        turn.done.wait()
        return np.concatenate(turn.chunks)

    def close(self):
        self._q.put(self._STOP)


def input_audio(tts: TTSWorker, item: dict) -> np.ndarray:
    """Kokoro-synthesized question, cached to disk, returned at 16 kHz."""
    AUDIO_DIR.mkdir(exist_ok=True)
    path = AUDIO_DIR / f"{item['id']}.wav"
    if path.exists():
        audio_24k, _ = sf.read(path, dtype="float32")
    else:
        audio_24k = tts.synth_blocking(item["text"])
        sf.write(path, audio_24k, TTS_SAMPLE_RATE)
    return resample_poly(audio_24k, 2, 3).astype(np.float32)


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------

class ASR:
    def __init__(self):
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self.repo, self.variant = whisper_repo()

    def transcribe(self, audio_16k: np.ndarray) -> str:
        result = self._mlx_whisper.transcribe(
            audio_16k, path_or_hf_repo=self.repo, verbose=None, language="en"
        )
        return result["text"].strip()

    def prewarm(self) -> float:
        """Load weights and compile the Metal graph on 1 s of silence."""
        t = time.monotonic()
        self.transcribe(np.zeros(16000, dtype=np.float32))
        return ms(t)


# --------------------------------------------------------------------------
# LLM paths. stream(messages) yields text fragments; stats() after the loop.
# --------------------------------------------------------------------------

class OllamaLLM:
    name = "ollama"
    model = OLLAMA_MODEL
    precision = "Q4_K_M"

    def __init__(self):
        import ollama

        self._ollama = ollama
        self.last: dict = {}

    def stream(self, messages: list[dict]):
        final = None
        for part in self._ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            think=False,
            keep_alive=-1,
            options={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS, "temperature": 0},
        ):
            final = part
            if getattr(part.message, "thinking", None):
                self.last["thinking_seen"] = True
            if part.message.content:
                yield part.message.content

        ns = lambda v: None if v is None else round(v / 1e6, 1)  # noqa: E731
        self.last.update(
            prompt_tokens=getattr(final, "prompt_eval_count", None),
            prompt_eval_ms=ns(getattr(final, "prompt_eval_duration", None)),
            load_ms=ns(getattr(final, "load_duration", None)),
            out_tokens=getattr(final, "eval_count", None),
            decode_tok_s=(
                round(final.eval_count / (final.eval_duration / 1e9), 1)
                if final and final.eval_duration else None
            ),
        )

    def unload(self):
        self._ollama.generate(model=OLLAMA_MODEL, prompt="", keep_alive=0)

    def load(self) -> float:
        """A real streamed turn with the same options as every later call.

        A bare `generate(prompt="")` loads with default num_ctx, and the first
        real call with num_ctx=4096 then reloads (measured: load_ms 1843 on
        the first turn after such a prewarm).
        """
        t = time.monotonic()
        for _ in self.stream([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": "Say ready."}]):
            pass
        return ms(t)

    def memory(self) -> dict:
        for m in self._ollama.ps().models:
            if m.model.startswith(OLLAMA_MODEL):
                return {"ollama_size_mb": round(m.size / 2**20),
                        "ollama_size_vram_mb": round(m.size_vram / 2**20)}
        return {}


class MlxLLM:
    name = "mlx"
    model = MLX_MODEL
    precision = "4bit (group 64)"

    def __init__(self):
        self.last: dict = {}
        self._model = None

    def load(self) -> float:
        t = time.monotonic()
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self._model, self._tok = load(MLX_MODEL)
        # One detokenizer for the session: `tokenizer.detokenizer` builds a
        # new one on every access (cycle 7: 154 ms -> 20 ms TTFT at 270m).
        self._detok = self._tok.detokenizer
        self._make_cache = make_prompt_cache
        # Compile the Metal graphs with one real turn, then start clean.
        self._reset_cache()
        for _ in self.stream([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": "Say ready."}]):
            pass
        self._reset_cache()
        return ms(t)

    def _reset_cache(self):
        self._cache = self._make_cache(self._model)
        self._cached: list[int] = []  # tokens whose KV/state is in the cache
        self._last_messages: list[dict] = []
        self._last_reply = ""

    def _encode(self, messages: list[dict]) -> list[int]:
        text = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return self._tok.encode(text, add_special_tokens=False)

    def _delta(self, messages: list[dict]) -> list[int] | None:
        """Tokens for the new user turn only, or None if not a continuation."""
        if not self._cached or len(messages) < 4 or messages[-2]["role"] != "assistant":
            return None
        if messages[-2]["content"] != self._last_reply or messages[:-2] != self._last_messages:
            return None
        rendered = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        marker = messages[-2]["content"] + "<|im_end|>"
        at = rendered.rfind(marker)
        if at < 0:
            return None
        return self._tok.encode(rendered[at + len(marker):], add_special_tokens=False)

    def stream(self, messages: list[dict]):
        import mlx.core as mx
        from mlx_lm.generate import generate_step, generation_stream, wired_limit

        # Qwen3.5 mixes attention and linear-attention layers; the latter's
        # ArraysCache cannot be trimmed. The chat template also renders the
        # generation prompt with an empty <think></think> block but drops it
        # from earlier assistant turns, so re-rendering the whole history
        # never matches the cache (measured: 0 hit tokens every turn).
        # Instead, extend the cache with only the new user turn, rendered by
        # the template itself. The model's context is then exactly what it
        # already saw, empty think blocks included.
        new = self._delta(messages)
        if new is not None:
            hit = len(self._cached)
        else:
            self._reset_cache()
            new, hit = self._encode(messages), 0

        self.last = {"prompt_tokens": hit + len(new), "cache_hit_tokens": hit,
                     "prefilled_tokens": len(new)}
        self._detok.reset()
        fed = list(new)
        tic = time.perf_counter()
        out = 0
        with wired_limit(self._model, [generation_stream]):
            for i, (token, _) in enumerate(generate_step(
                mx.array(new), self._model, max_tokens=MAX_TOKENS,
                prompt_cache=self._cache,
            )):
                token = int(token)
                # generate_step schedules the model call on a token before
                # yielding it, so every yielded token (EOS included) is
                # already in the cache.
                fed.append(token)
                if i == 0:
                    self.last["prompt_eval_ms"] = round((time.perf_counter() - tic) * 1000, 1)
                    tic = time.perf_counter()
                if token in self._tok.eos_token_ids:
                    break
                out += 1
                self._detok.add_token(token)
                seg = self._detok.last_segment
                if seg:
                    yield seg
            self._detok.finalize()
            tail = self._detok.last_segment
            if tail:
                yield tail

        elapsed = time.perf_counter() - tic
        self._cached = self._cached + fed if hit else fed
        self._last_messages = list(messages)
        self._last_reply = self._detok.text.strip()
        self.last.update(
            out_tokens=out,
            decode_tok_s=round(out / elapsed, 1) if elapsed and out else None,
        )

    def unload(self):
        pass

    def memory(self) -> dict:
        import mlx.core as mx

        return {"mlx_active_mb": round(mx.get_active_memory() / 2**20),
                "mlx_peak_mb": round(mx.get_peak_memory() / 2**20)}


# --------------------------------------------------------------------------
# One turn under one clock
# --------------------------------------------------------------------------

def run_turn(asr: ASR, llm, tts: TTSWorker, history: list[dict], item: dict,
             audio_16k: np.ndarray, *, phase: str, round_: int) -> dict:
    t0 = time.monotonic()  # end of speech

    transcript = asr.transcribe(audio_16k)
    stt_ms = ms(t0)

    history.append({"role": "user", "content": transcript})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    turn = TurnAudio(started=t0)
    segmenter = Segmenter()
    reply = []
    llm_start = time.monotonic()
    ttft_ms = first_sentence_ms = None

    for fragment in llm.stream(messages):
        if ttft_ms is None:
            ttft_ms = ms(llm_start)
        reply.append(fragment)
        for sentence in segmenter.feed(fragment):
            if first_sentence_ms is None:
                first_sentence_ms = ms(t0)
            tts.say(turn, sentence)
    llm_total_ms = ms(llm_start)
    for sentence in segmenter.flush():
        if first_sentence_ms is None:
            first_sentence_ms = ms(t0)
        tts.say(turn, sentence)
    tts.end(turn)
    turn.done.wait()
    end_to_end_ms = ms(t0)

    reply_text = "".join(reply).strip()
    history.append({"role": "assistant", "content": reply_text})

    audio = np.concatenate(turn.chunks) if turn.chunks else np.zeros(1, np.float32)
    wav = AUDIO_DIR / f"{llm.name}-{phase}-r{round_}-{item['id']}__reply.wav"
    sf.write(wav, audio, TTS_SAMPLE_RATE)

    return {
        "path": llm.name,
        "model": llm.model,
        "precision": llm.precision,
        "phase": phase,
        "round": round_,
        "turn_index": len(history) // 2,
        "id": item["id"],
        "stt_ms": stt_ms,
        "llm_ttft_ms": ttft_ms,
        "first_sentence_ms": first_sentence_ms,
        "time_to_first_audio_ms": turn.first_audio_ms,
        "tts_first_audio_ms": (
            round(turn.first_audio_ms - first_sentence_ms, 1)
            if turn.first_audio_ms and first_sentence_ms else None
        ),
        "llm_total_ms": llm_total_ms,
        "tts_synth_total_ms": round(turn.synth_ms, 1),
        "end_to_end_ms": end_to_end_ms,
        "reply_audio_s": round(len(audio) / TTS_SAMPLE_RATE, 2),
        **llm.last,
        "transcript": transcript,
        "reply_text": reply_text,
        "reply_hash": text_hash(reply_text),
        "correct": is_correct(reply_text, item),
        "reply_wav": str(wav.relative_to(HERE)),
    }


def run_round(asr, llm, tts, items, audio, *, phase: str, round_: int) -> list[dict]:
    """One append-only conversation over all items."""
    history: list[dict] = []
    return [
        run_turn(asr, llm, tts, history, item, audio[item["id"]], phase=phase, round_=round_)
        for item in items
    ]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

METRICS = ["stt_ms", "llm_ttft_ms", "first_sentence_ms", "time_to_first_audio_ms",
           "llm_total_ms", "end_to_end_ms", "decode_tok_s"]


def summarize(turns: list[dict], path: str) -> dict:
    warm = [t for t in turns if t["path"] == path and t["phase"] == "warm"]
    out = {"path": path, "n": len(warm)}
    for m in METRICS:
        vals = [t[m] for t in warm if t.get(m) is not None]
        if not vals:
            continue
        # spread: median over items of (max - min across that item's repeats),
        # so item-to-item variation is not counted as noise
        per_item = {}
        for t in warm:
            if t.get(m) is not None:
                per_item.setdefault(t["id"], []).append(t[m])
        out[m] = round(median(vals), 1)
        out[m + "_spread"] = round(median(max(v) - min(v) for v in per_item.values()), 1)
    by_item = {}
    for t in warm:
        by_item.setdefault(t["id"], []).append(t["correct"])
    out["items_correct"] = sum(all(v) for v in by_item.values())
    out["items_total"] = len(by_item)
    out["turns_correct"] = sum(t["correct"] for t in warm)
    return out


def verdict(summary: dict[str, dict], tfa_budget_ms: float = 1500, min_correct: int = 5) -> dict:
    per_path = {
        p: {
            "time_to_first_audio_ms": s["time_to_first_audio_ms"],
            "tfa_under_budget": s["time_to_first_audio_ms"] < tfa_budget_ms,
            "items_correct": f"{s['items_correct']}/{s['items_total']}",
            "correct_ok": s["items_correct"] >= min_correct,
        }
        for p, s in summary.items()
    }
    claim_holds = all(v["tfa_under_budget"] and v["correct_ok"] for v in per_path.values())

    a, b = summary["ollama"], summary["mlx"]
    gap = abs(a["llm_ttft_ms"] - b["llm_ttft_ms"])
    noise = max(a["llm_ttft_ms_spread"], b["llm_ttft_ms_spread"])
    winner = None
    if gap > noise:
        winner = "ollama" if a["llm_ttft_ms"] < b["llm_ttft_ms"] else "mlx"
    return {
        "tfa_budget_ms": tfa_budget_ms,
        "min_items_correct": min_correct,
        "per_path": per_path,
        "claim_holds": claim_holds,
        "ttft_gap_ms": round(gap, 1),
        "ttft_noise_ms": noise,
        "ttft_winner": winner or "none (gap within spread)",
        "endpointing_ms_not_included": ENDPOINTING_MS,
    }


def save(results: dict):
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
