# Sub-second local voice turns on a 4B multimodal model

**Serving path, cache reuse and prewarm for ASR → LLM → TTS on Apple M4**

Date: 2026-09-16. Evidence: `results.json`, `notebook.ipynb`, `pipeline.py` in this folder.
Hardware: MacBook Air M4, 10-core GPU, 24GB unified memory.

## Abstract

- **Setup:** a local voice turn (mlx-whisper small, then qwen3.5:4b, then Kokoro)
  measured under one monotonic clock, with every latency optimization known so far
  applied.
- **Paths compared:** Ollama over HTTP (Q4_K_M) and in-process mlx-lm (4-bit).
- **Latency:** warm median time to first audio from end of speech is 860ms on mlx-lm
  and 1872ms on Ollama.
- **Correctness:** 8/8 short factual items on both paths.
- **Cause of the gap:** not raw decode speed. Ollama does not reuse the prompt prefix for
  this hybrid-attention model, so its TTFT grows linearly with conversation length.
  mlx-lm stays flat at ~135ms once one template quirk is worked around.
- **Prewarm:** removes a 1.4s first-turn penalty, but only when it uses the exact
  request options of real turns.

## 1. Problem

- **Latency budget:** a conversational reply feels natural when audio starts within
  roughly one second of the user finishing.
- **Earlier optimizations were measured one stage at a time:**
  - mlx-whisper on Metal
  - Kokoro on MPS
  - a smaller LLM
  - append-only history
  - a cached detokenizer
- **The quoted 2.3s "optimized total" was a sum of medians** from separate runs, never
  one turn.
- **The small model that made it fast** (gemma3:270m) answered 2 of 6 facts wrong.
- **Open question:** can a 4B-class model with tools and vision fit a sub-1.5s budget
  once every known optimization is applied together?

## 2. Method

### 2.1 Pipeline

```
input WAV (in memory) ─► mlx-whisper small.en + accent LoRA
                      ─► LLM stream (Ollama | mlx-lm), thinking off
                      ─► sentence segmenter ─► Kokoro worker thread (MPS) ─► WAV
```

- **Clock start:** when the input audio is handed to ASR, i.e. end of speech.
- **Headline metric:** time to first audio, meaning the first sentence has been
  synthesized and its padding trimmed.
- **Excluded:** endpointing (1.5s in the live agents) and speaker output latency.

### 2.2 Paths under test

| | Ollama | mlx-lm |
|---|---|---|
| Model | `qwen3.5:4b` | `mlx-community/Qwen3.5-4B-4bit` |
| Precision | Q4_K_M (4.7B incl. vision) | 4-bit affine, group 64 |
| Transport | HTTP to local server | In-process |
| Cache | Server-managed | Persistent `make_prompt_cache`, extended per turn |

### 2.3 Optimizations applied to both paths

| Stage | Optimization | Prior evidence |
|---|---|---|
| ASR | mlx-whisper on Metal instead of CPU faster-whisper | 7.55x (`compare_stt.py`) |
| ASR | Accent LoRA checkpoint | earlier accent spike |
| LLM | Thinking disabled; TTFT = first content token | qwen3:4b: 139s to content |
| LLM | Append-only history | KV prefix reuse on gemma3 (cycle 7) |
| LLM | 200-token cap, temperature 0, one-sentence system prompt | Alice config |
| TTS | Kokoro on MPS, one long-lived worker thread | 2.2x (`compare_tts.py`) |
| TTS | Sentence streaming (`segment.py`), padding trim | `tts.py` |
| All | Prewarm every stage with a real pass | step0 finding |

### 2.4 Protocol

- **Items:** 8 `fact-*` items from `data/dataset.jsonl`, spoken by Kokoro
  `af_heart`.
- **Rounds:** 3 warm rounds per path. Each round is one append-only conversation over all
  8 items, and path order alternates between rounds.
- **Correctness:** `expect.contains_any` on the reply text.
- **Spread:** median over items of (max − min) across that item's 3 repeats, so
  item-to-item variation does not count as noise.
- **A/B rule:** a winner is named only when the gap exceeds the larger spread.

## 3. Results

### 3.1 Warm medians (24 turns per path)

| Metric | Ollama | mlx-lm | Ratio |
|---|---:|---:|---:|
| STT | 214ms | 214ms | 1.0x |
| LLM TTFT | 714 ±34ms | 136 ±3ms | 5.3x |
| LLM total | 1198ms | 382ms | 3.1x |
| First sentence ready | 1465ms | 651ms | 2.3x |
| **Time to first audio** | **1872 ±500ms** | **860 ±13ms** | **2.2x** |
| Decode | 20.7 tok/s | 35.7 tok/s | 1.7x |
| Items correct | 8/8 | 8/8 | |

- **Claim** (under 1500ms and at least 5 items correct): holds on mlx-lm, fails on
  Ollama.
- **A/B:** TTFT gap 578ms against 34ms of noise, so mlx-lm wins.

### 3.2 TTFT against conversation length (round 2)

| Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ollama TTFT (ms) | 442 | 500 | 597 | 619 | 746 | 854 | 1003 | 1118 |
| Ollama prompt eval (ms) | 335 | 396 | 486 | 504 | 634 | 743 | 891 | 1003 |
| mlx-lm TTFT (ms) | 220 | 135 | 136 | 134 | 135 | 136 | 136 | 135 |
| mlx-lm cached tokens | 0 | 57 | 82 | 115 | 142 | 175 | 207 | 239 |

### 3.3 Startup and memory

| Item | Value |
|---|---:|
| Prewarm total, all stages | 25.3s |
| mlx-whisper + LoRA load and warm | 12.4s |
| Ollama load and warm turn | 3.9s |
| mlx-lm load and warm turn | 3.1s |
| Ollama model resident (`num_ctx` 4096) | 5572MB |
| mlx-lm active / peak | 2785 / 3492MB |

## 4. Analysis

### 4.1 The bottleneck is prefill, not decode

- **Ollama's time is in prompt evaluation.**
  - Its `prompt_eval_duration` grows ~90ms per turn.
  - An identical 49-token prompt re-evaluates in 256ms on every repeat, so nothing is
    reused.
- **qwen3.5 mixes attention and linear-attention (recurrent) layers.**
  - Recurrent state cannot be sliced back to an arbitrary prefix, the way a KV cache
    can.
  - A server that matches prefixes by truncating the cache has nothing to truncate to.
  - This is the likely cause. It was not confirmed in Ollama's source.
- **gemma3 is pure attention**, and on the same server its prompt evaluation stayed flat
  (cycle 7).
- **Decode speed matters less.** 1.7x faster decode on mlx-lm is secondary for
  one-sentence replies of 5–11 tokens.

### 4.2 Hybrid models need append-only cache discipline

- **In-process, the cache is owned by the caller.**
- **The Qwen3.5 chat template defeats naive reuse.**
  - The generation prompt ends in an empty `<think>\n\n</think>\n\n`.
  - Past assistant turns are re-rendered without it.
  - Re-rendering the full history and comparing token prefixes matched 0 tokens on every
    turn.
- **The recurrent cache cannot be trimmed**, so a mismatch forces a full re-prefill.
- **Working rule:**
  - Never re-render history.
  - Append only the new user turn, rendered by the template itself, to the tokens already
    in the cache.
  - The model's context then equals exactly what it has already processed.
  - Cost: past turns keep their empty think blocks, which a canonical re-render would
    drop.

### 4.3 Prewarm must replay the real request

| Prewarm mistake | Measured cost on first real turn |
|---|---|
| Ollama load via bare `generate(prompt="")` | Reload when `num_ctx` differs: 1843ms |
| Single Kokoro warm pass on MPS | Second call still 927ms; third 165ms |
| No LLM warm turn on mlx-lm | First TTFT 1410ms vs 221ms |

- **Rule:** prewarm with a full turn that uses the production options (context size,
  token cap, thinking flag), and run TTS twice.

### 4.4 Remaining budget at 860ms

| Component | ms | Lever |
|---|---:|---|
| ASR | 214 | Streaming ASR; smaller encoder |
| LLM TTFT | 136 | Already cache-bound |
| Rest of decode to sentence end | ~250 | Shorter first clause; speak before sentence end |
| Segmenter hold-back + spaCy | ~50 | Release single-sentence replies at EOS token |
| Kokoro first sentence | ~200 | Pre-synthesized fillers; shorter first chunk |

- **Next step:** after ASR, the largest single cost is waiting for a complete sentence.
  That motivates a parallel fast track that speaks a short filler or backchannel while
  the main model is still decoding.

## 5. Limitations

- **Workload:**
  - One machine, one run, batch 1.
  - Short factual prompts only.
  - Single-sentence replies, so sentence streaming was not exercised.
- **Quality:** checked only by substring match on 8 items. Cross-path text matched on 7
  of 8.
- **Scope:** tool calling, vision, RAG and multi-agent memory pressure were not measured.
- **Unexplained TTS slowdown:** in one round, Kokoro slowed to 710–870ms per sentence,
  which accounts for all of Ollama's ±500ms spread. Not investigated.
- **Input:** synthesized speech, not live microphone audio.

## 6. Recommendations

1. **Serve qwen3.5-class hybrid models in-process with mlx-lm** for voice. Don't use
   Ollama while it lacks prefix reuse for them.
2. **Keep the prompt cache append-only.** Build each turn's delta from the template, and
   never diff a re-rendered history.
3. **Prewarm with production request options,** and run two TTS passes.
4. **Measure every change as one turn under one clock.** Summed medians across runs hid
   the real bottleneck.
5. **Attack the ~500ms between LLM TTFT and first audio** with a parallel filler track.
   That is the follow-up spike.
