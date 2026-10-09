# Multi-track voice: 270m filler + 1b speaker + qwen3.5 reasoner

Three mlx 4-bit models run as parallel tracks in one process to fill the silence before
a 4B model answers.

| Track | Model | Job |
|---|---|---|
| Filler | `gemma-3-270m-it-qat-4bit` | Picks a filler or backchannel phrase from a pre-synthesized bank |
| Speaker | `gemma-3-1b-it-qat-4bit` | Writes the spoken answer, streamed into Kokoro |
| Reasoner | `Qwen3.5-4B-4bit` | Tools (time, calculator, doc search); first token routes the turn; answers go to the shared context |

Baseline from experiment `260917-voice-pipeline-qwen35`: one qwen3.5 model reached 860ms warm time to
first audio.

## Claim and falsifier

With all three models preloaded and prewarmed and Ollama unloaded:

| # | Check | Threshold |
|---|---|---|
| 1 | Warm median first audio after end of speech | ≤ 400ms |
| 2 | Plain fact answers start | ≤ 1000ms |
| 3 | Routing (backchannel vs answer) correct | ≥ 80% |
| 4 | Speaker TTFT when combined vs alone | ≤ 2x |

**Falsifier:** any check fails.

## Outcome (2026-09-16, MacBook Air M4, 24GB)

| Policy | Checks | Verdict |
|---|---|---|
| A `always_filler` | 1 and 3 pass, 2 and 4 fail | Claim fails |
| B `route_aware` | All 4 pass | **Claim holds** |

Warm medians, 15 items × 3 rounds = 45 turns per policy:

| Metric | A `always_filler` | B `route_aware` |
|---|---:|---:|
| STT (mlx-whisper + accent LoRA) | 213ms | 214ms |
| Filler decision (270m) | 95ms | 95ms |
| **First audio, all turns** | 313ms | 373ms |
| First audio, backchannel turns | 309ms | 314ms |
| Fact answer ready | 764ms | 616ms |
| **Fact answer starts playing** | **1041ms** | **616ms** |
| Speaker TTFT (1b) | 134ms | 34ms |
| Tool routing signal (qwen first token) | 361ms | 363ms |
| Tool answer starts playing | 3174ms | 2899ms |
| Routing correct | 39/45 | 39/45 |
| Facts correct | 21/24 | 21/24 |
| Tool answers correct | 6/9 | 6/9 |

Both policies share the same missed facts and tool answers, all caused by routing
misses.

### Policies

- **A `always_filler`:**
  - Every turn plays the 270m's phrase as soon as it decides.
  - The reasoner always runs to completion.
- **B `route_aware`:**
  - A backchannel plays immediately and ends the turn.
  - A hold phrase plays only when the reasoner's first token is `<tool_call>`.
  - Without a tool call the reasoner is cancelled, and the speaker answers with no
    filler.

### Each model alone vs combined

`bench_independent.py` runs each model alone in a fresh process, with no ASR or TTS.

| Model | Alone: TTFT / decision | Combined A | Combined B | Memory active / peak |
|---|---:|---:|---:|---:|
| 270m filler (pinned few-shot prefix) | 18ms decision, 11ms TTFT | 95ms (5.3x) | 95ms (5.3x) | 245 / 339MB |
| 1b speaker (append-only cache) | 35ms TTFT, 146 tok/s | 134ms (3.9x) | 34ms (1.0x) | 717 / 767MB |
| qwen3.5 reasoner (pinned tool prompt) | 147ms TTFT | 145ms | 150ms | 2329 / 3175MB |

**All three loaded together:**

| Item | Value |
|---|---:|
| mlx active memory | 3793MB |
| mlx peak memory | 4577MB |
| Python process RSS (includes Whisper and Kokoro) | 6.5GB |
| Prewarm, whole stack | 25–26s |

**What each model does alone:**

| Model | Result |
|---|---|
| Reasoner | Tool answers 9/9 correct; facts 24/24; tool turns total 1289ms |
| Speaker | Facts 21/24; tool questions 0/9 (it guesses, e.g. 17×23 → "One hundred eighty-nine") |
| Filler | Routing 42/45 |

## Findings

1. **Contention decides the design.**
   - The reasoner prefills at the start of every turn.
   - If it keeps decoding a plain-fact answer nobody will use, the 1b speaker's TTFT
     rises 3.9x.
   - Cancelling the reasoner once its first token shows no tool call brings the speaker
     back to its solo 34ms.
   - The 270m still pays 5.3x because it runs at the same moment as the reasoner's
     prefill. It still decides in 95ms.
2. **A filler can delay the answer.**
   - A clip, once playing, can't be cut mid-word.
   - Under A, fact answers were ready at 764ms but started at 1041ms.
   - Hold fillers only pay off when the wait is long, i.e. tool turns (1.8–4.0s).
3. **A small model can't route itself; the big model's first token can.**

   | Router | Result |
   |---|---|
   | 1b told to say CHECKING | 0/9 tool questions (guessed wrong) |
   | 1b with few-shot CHECKING examples | Deferred 6/8 plain facts |
   | qwen `<tool_call>` first token | 9/9 tool questions in ~150ms |

   - The qwen router also sends 2 of 8 facts to tools: calculate for "days in a leap
     year", search_docs for "Mount Fuji". Both answers were correct, but slower
     (1.6–2.0s).
4. **The 270m is a weak router.**

   | Prompt style | Result |
   |---|---|
   | Single-message list of phrases | "Okay." on 15/15 items |
   | Chat-format few-shot | 14/15 correct |

   - With chat-format few-shot it often answers the question itself ("Paris.",
     "Blue."). The matcher sends anything that isn't a known phrase to hold.
   - It fails the same two items in every round:
     - "who wrote Romeo and Juliet." (lower case, no question mark) → "Yes."
     - "What time is it right now?" → "Okay."
   - All 6 routing misses are these, and they account for every wrong fact and tool
     answer in both policies.
5. **Fixed prompts get pinned prefixes.**
   - 270m: the few-shot prefix is prefilled once, and each call trims back to it
     (18ms decision).
   - Qwen3.5: the hybrid cache can't be trimmed, so the ~400-token tool prompt is
     restored from a copied snapshot (~0.3ms).
   - Reasoner TTFT drops 1226ms → 147ms, and the tool call is byte-identical.
6. **gemma's tokenizer lists only `<eos>` as a stop token.** Turns end on
   `<end_of_turn>`, which has to be added.
7. **mlx's `detokenizer.last_segment` consumes what it returns.**
   - Reading it twice per step (once to check, once to pass on) silently dropped all
     streamed text.
   - The speaker produced correct replies that never reached TTS.
8. **Threads are fine.** Two mlx models generating concurrently from two threads ran
   without errors on mlx 0.31.3.
9. **The phrase bank removes TTS time for fillers.**
   - Live synthesis of the same phrases: 168–170ms median.
   - Clips are synthesized at speed 1.15 with 40ms edge padding.
   - They still last 395–905ms, which is why they must not sit in front of a
     ready answer.

## Not measured

- **Live microphone, endpointing, playback:** input is pre-synthesized audio.
- **Barge-in / interruption behaviour.**
- **Filler naturalness:** no listening test beyond the saved WAVs.
- **RAG over embeddings.** `search_docs` is a keyword index over the qwen spike's docs,
  so Ollama stays unloaded.
- **Reasoner thinking mode:** off.
- **Qwen3.5 vision.**
- **Routing on real conversational speech:** 15 scripted items only.

## Run

Self-contained: needs only the files in this folder. Apple silicon (MLX), Python 3.12.

```bash
brew install espeak-ng
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./verify_multitrack.sh                       # checks, independent x3, both policies
./verify_multitrack.sh --skip-independent
MULTITRACK_POLICY=always_filler .venv/bin/python -m jupyterlab notebook.ipynb
```

Models download from HF on first load:

- `gemma-3-270m-it-qat-4bit`
- `gemma-3-1b-it-qat-4bit`
- `Qwen3.5-4B-4bit`

## Files

| File | Contents |
|---|---|
| `qwen_pipeline.py`, `segment.py`, `tts.py` | ASR + helpers from the qwen3.5 experiment, sentence segmenter, Kokoro helpers |
| `reference/` | qwen3.5 experiment docs, the corpus `search_docs` indexes |
| `data/dataset.jsonl` | Evaluation prompts |
| `requirements.txt` | Python dependencies |
| `tracks.py` | Model wrapper (cache, pinning), TTS + bank, filler / speaker / reasoner tracks, turn orchestration, policies, summary |
| `bench_independent.py` | One model alone, fresh process → `results_independent.json` |
| `notebook.ipynb` | Combined run, `route_aware` outputs |
| `notebook_always_filler.ipynb` | Same notebook, `always_filler` outputs |
| `verify_multitrack.sh` | Checks, Ollama unload, independent benches, both policies, consistency |
| `results_*.json` | Every turn: transcript, filler raw output, route, tool, timeline, reply |
| `audio/` | Input and reply WAVs (gitignored) |
