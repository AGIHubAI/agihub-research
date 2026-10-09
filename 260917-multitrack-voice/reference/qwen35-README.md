# Voice pipeline on qwen3.5:4b: ASR → LLM → TTS, every known optimization

One notebook runs the full voice turn under one monotonic clock. It combines every
latency optimization found so far, plus prewarm, on qwen3.5:4b (4GB class,
Ollama capabilities `tools`, `vision`, `thinking`). The LLM stage is A/B tested over
two serving paths.

## Claim and falsifier

- **Claim:** with prewarm and every known optimization, the warm median time to first
  audio stays under 1500ms from end of speech. Answers are correct on at least 5 items.
- **Falsifier:** either condition fails on either LLM path.
- **A/B rule:** no winner is named when the TTFT gap is inside the per-item spread.
- **Items:** the plan said 6 `fact-*` items. `data/dataset.jsonl` has 8, and all 8 are
  used.

## Outcome (2026-09-16, MacBook Air M4, 24GB)

**The claim holds on mlx-lm and fails on Ollama.** Both paths answer 8/8 items
correctly on every round. mlx-lm wins TTFT by 578ms against 34ms of noise.

Warm medians: 8 items × 3 rounds per path = 24 turns. Spread is the median over items of
max − min across that item's 3 repeats.

| Stage (ms) | Ollama `qwen3.5:4b` Q4_K_M | mlx-lm `Qwen3.5-4B-4bit` |
|---|---:|---:|
| STT (mlx-whisper + accent LoRA) | 214 ±7 | 214 ±7 |
| LLM TTFT | 714 ±34 | **136 ±3** |
| LLM total | 1198 ±48 | 382 ±13 |
| First sentence ready | 1465 ±50 | 651 ±11 |
| **Time to first audio** | **1872 ±500** | **860 ±13** |
| Decode (tok/s) | 20.7 | 35.7 |
| Items correct | 8/8 | 8/8 |

Not included: endpointing (fixed at 1500ms in the live agents) and speaker output
latency. Most replies are one sentence, so time to first audio equals end to end.

Prewarm, paid once at startup:

| Stage | ms |
|---|---:|
| Kokoro load + 2 warm synths | 4423 + 1330 |
| mlx-whisper (accent LoRA from T7) load + 1 s silence | 12389 |
| Ollama load (after explicit unload) + 1 warm turn | 3932 |
| mlx-lm load + 1 warm turn | 3092 |
| **Total** | **25303** |

Memory, partial. It covers this Python process and Ollama's model, not a whole voice room:

| Process / model | MB |
|---|---:|
| Python RSS peak (Whisper + Kokoro + mlx-lm) | 1606 |
| mlx-lm active | 2785 |
| mlx-lm peak | 3492 |
| Ollama `qwen3.5:4b` loaded, `num_ctx` 4096, including vision | 5572 |

The Python RSS peak probably does not count Metal buffers. The mlx-lm figures are
reported separately.

## Findings

1. **Ollama does not reuse the prefix cache for qwen3.5.**
   - `prompt_eval_ms` grows with every turn: 361 → 951ms from turn 1 to turn 8.
   - An identical 49-token prompt re-evaluated in 256ms on every repeat.
   - TTFT therefore climbs with conversation length, 474 → 1081ms.
   - This is the opposite of gemma3, where `prompt_eval` stayed flat at ~88ms (cycle 7).
   - Likely cause: qwen3.5 is a hybrid attention / linear-attention model. Not
     confirmed in Ollama's source.
2. **mlx-lm TTFT is flat at ~135ms from turn 2 on.** It prefills only the 19–22 new
   tokens, with 57 → 239 tokens reused from the cache.
3. **The Qwen3.5 chat template breaks naive cache reuse.**
   - The generation prompt ends with an empty `<think>\n\n</think>\n\n`.
   - Earlier assistant turns are re-rendered without it.
   - Re-rendering the full history matched 0 cached tokens every turn.
   - The linear-attention `ArraysCache` can't be trimmed, so rollback isn't possible.
   - Fix used here: extend the cache with only the new user turn, rendered by the
     template (`MlxLLM._delta`). The model's context keeps the empty think blocks it
     actually saw, so it differs from a canonical re-render by those blocks.
4. **Decode is 1.7x faster on mlx-lm** (35.7 vs 20.7 tok/s) at 4-bit on both paths.
   It matches cycle 7's direction at 270m.
5. **Prewarm has to use the real request options.**
   - A bare Ollama `generate(prompt="")` loads with the default `num_ctx`.
   - The first real call with `num_ctx=4096` then reloaded: `load_ms` 1843.
   - Kokoro on MPS needs 2 passes: the first synth took 927ms, the second 165ms.
   - In a smoke test outside the notebook, the first mlx-lm TTFT was 1410ms without
     a prewarm turn and 221ms with one.
6. **The first turn of each new conversation is slower on mlx-lm** (220–470ms TTFT).
   It has a full prefill and no cache. The notebook starts a fresh conversation every
   round, so 3 of the 24 warm turns carry this.
7. **The Ollama spread (±500ms) comes from TTS, not the LLM.**
   - In Ollama's round 1, Kokoro took 710–870ms for turns 3–8.
   - Everywhere else it took ~175–300ms.
   - One mlx-lm turn (round 1, fact-006) showed the same thing.
   - Not explained. It may be GPU contention with the other engine's memory or thermal
     state. Not investigated.
8. **Sentence streaming gains nothing on one-sentence replies.**
   - `Segmenter` holds back the last sentence until the stream ends.
   - A one-sentence reply is released only after the full decode (`first_sentence_ms`
     ≈ STT + LLM total + ~50ms).
   - The gain applies to multi-sentence replies, which this item set does not produce.

## Optimizations applied

Recorded as flags in `results.json` `meta.optimizations`:

| Stage | Optimization |
|---|---|
| ASR | mlx-whisper on Metal |
| ASR | Accent LoRA checkpoint from T7 |
| ASR | Prewarm transcribe on 1 s of silence |
| LLM | Thinking off |
| LLM | Streaming, TTFT measured to the first content token |
| LLM | Append-only history, one conversation per round |
| LLM | Token cap of 200 |
| LLM | Short spoken-style system prompt |
| LLM | Temperature 0 on both paths |
| Ollama | `keep_alive=-1` |
| Ollama | Fixed `num_ctx=4096` on every call, prewarm included |
| mlx-lm | In-process |
| mlx-lm | One cached detokenizer |
| mlx-lm | Persistent prompt cache extended per turn |
| mlx-lm | Prewarm turn |
| TTS | Kokoro `device="mps"` |
| TTS | Sentence streaming into one long-lived Kokoro worker thread (`segment.Segmenter`) |
| TTS | `tts.trim_padding` |
| TTS | Two warm synths |

## Not measured

- **Endpointing and VAD:** input is pre-synthesized Kokoro audio held in memory.
- **Playback latency.**
- **Tool calling and vision.** The capabilities are recorded, not exercised.
- **RAG.**
- **Concurrent voice room:** DEXA, Alice and Nova were not running.
- **Answer quality beyond `contains_any`** on 8 short factual items.
  - Each path gave the same text on all 3 rounds of every item.
  - Across paths, text matched on 7 of 8 items. fact-006 was worded differently, and
    both versions were correct.
- **Multi-sentence replies.**

## Run

Self-contained: needs only the files in this folder. Tested versions: mlx_whisper 0.4.3,
mlx_lm 0.31.3, kokoro 0.9.4, ollama 0.6.2. Apple silicon (MLX), Python 3.12.

```bash
brew install espeak-ng ollama
ollama pull qwen3.5:4b                       # 3.4GB, once
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./verify_pipeline.sh                         # checks + headless run, ~3 min

# interactive
.venv/bin/python -m jupyterlab notebook.ipynb
```

`mlx-community/Qwen3.5-4B-4bit` (~2.5GB) downloads from HF on first load.
Optional: `ACCENT_LORA_DIR=<dir with weights.safetensors>` uses an accent-tuned Whisper;
the weights are not published, so the default is the public `whisper-small.en`.

## Files

| File | Contents |
|---|---|
| `segment.py`, `tts.py` | Sentence segmenter and Kokoro synthesis helpers |
| `data/dataset.jsonl` | Evaluation prompts; this experiment uses the 8 `fact-*` items |
| `requirements.txt` | Python dependencies |
| `pipeline.py` | ASR, the two LLM paths, the TTS worker, `run_turn`, summary and verdict |
| `notebook.ipynb` | Prewarm → input audio → first turn → warm rounds → memory → verdict → listen |
| `verify_pipeline.sh` | Environment checks, headless run, `results.json` consistency check |
| `results.json` | Every turn, with transcript, reply text, hash and per-stage timings |
| `audio/` | Input and reply WAVs, gitignored |
