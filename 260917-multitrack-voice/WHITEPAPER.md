# Parallel model tracks for local voice turns

**A 270M filler, a 1B speaker and a 4B reasoner on one Apple M4**

Date: 2026-09-16.

- **Evidence:** in this folder (`results_independent.json`,
  `results_combined_{always_filler,route_aware}.json`, `tracks.py`, notebooks).
- **Predecessor:** experiment `260917-voice-pipeline-qwen35` (`reference/qwen35-WHITEPAPER.md`).
- **Hardware:** MacBook Air M4, 10-core GPU, 24GB unified memory.

## Abstract

- **Baseline:** one in-process qwen3.5-4B pipeline reaches first audio at 860ms after
  end of speech.
- **Question:** can smaller models fill that gap without the 4B's tool use and reasoning
  being lost?
- **Setup:** three mlx 4-bit models run as parallel tracks in one process, on one GPU.
  - A 270M model picks a filler or backchannel from a pre-synthesized phrase bank.
  - A 1B model writes the spoken answer.
  - A 4B model calls tools and writes results to a shared context. Its first generated
    token decides whether the turn needs a tool.
- **Result:** with a route-aware playback policy, all three preloaded, and Ollama
  unloaded:
  - first audio at 373ms median (backchannels 314ms)
  - plain factual answers start at 616ms, 28% faster than the single 4B model
  - tool answers are preceded by a filler at ~370–430ms
- **Failed alternative:** playing a filler on every turn plus letting the 4B finish its
  answer. Fillers push ready answers to 1041ms, and GPU contention slows the 1B speaker
  3.9x.
- **Main lesson:** parallel tracks pay off only when unused work is cancelled early and
  fillers are reserved for waits long enough to need them.

## 1. Problem

The predecessor spike brought a local 4B voice turn to 860ms:

| Component | ms |
|---|---:|
| ASR | 214 |
| LLM TTFT | 136 |
| Decode to sentence end | ~250 |
| Kokoro first sentence | ~200 |

- **Two costs remain:**
  - silence while the model decodes
  - the 4B model's latency on work that needs no 4B
- **Human conversation fills both.** Short turns get backchannels ("I see.",
  "Of course."). Longer thinking gets a hold phrase ("Let me check that."). Easy
  questions get a fast answer.
- **The design question:** can small local models take those roles while the 4B model
  keeps tools and reasoning, within one laptop's memory and one GPU?

## 2. Architecture

```
                     ┌──────────────────────────────────────────────┐
 speech ─► ASR ─────►│ t=0 (end of ASR, ~214 ms after end of speech)│
 (mlx-whisper)       └──────────────────────────────────────────────┘
                        │                     │
                        ▼                     ▼
              Filler (270m, 4-bit)   Reasoner (qwen3.5-4B, 4-bit, tools)
              pinned few-shot prefix pinned tool-schema prefix
              phrase in ~95 ms       first token in ~150 ms
                        │                     │
            backchannel?│                     │ <tool_call>?
             ├─ yes ────┼─► play bank clip,   │
             │          │   cancel reasoner   │
             └─ no ─────┴─────────────────────┤
                                   ┌──────────┴──────────┐
                              tool │                     │ no tool
                                   ▼                     ▼
                     play hold clip (bank)      cancel reasoner
                     run tools, write result    Speaker (1b) answers,
                     to shared context          streamed to Kokoro
                     Speaker (1b) voices result
```

### 2.1 Components

| Track | Model | Role | Key optimization |
|---|---|---|---|
| Filler | `gemma-3-270m-it-qat-4bit` | Choose a phrase from an 11-entry bank | Few-shot prefix prefilled once; trimmed back after each call |
| Speaker | `gemma-3-1b-it-qat-4bit` | Spoken answer | Append-only prompt cache; streamed segments |
| Reasoner | `Qwen3.5-4B-4bit` | Tool calls (time, calculator, doc search), routing | 400-token tool prompt held as a copied cache snapshot |
| TTS | Kokoro on MPS | Answers live; fillers from bank | Bank pre-synthesized at speed 1.15, 40ms padding |
| ASR | mlx-whisper small.en + accent LoRA | Transcript | Prewarmed |

- **One process:** all models live in a single process.
- **Threads:** each track runs on its own thread, and each model is guarded by a lock.
  Concurrent generation from two threads was verified safe on mlx 0.31.3.

### 2.2 Playback policies

| Policy | Filler playback | Reasoner without a tool call |
|---|---|---|
| A `always_filler` | Every turn, as soon as the 270m decides | Runs to completion |
| B `route_aware` | Backchannels at once; hold phrases only on tool turns | Cancelled at its first token |

## 3. Method

- **Items:** 15 per round.

  | Type | Count | Source | Expected route |
  |---|---:|---|---|
  | Factual questions | 8 | `evals/golden/dataset.jsonl` | answer |
  | Remarks | 4 | "Thanks, that was really helpful." | backchannel |
  | Tool questions | 3 | arithmetic, current time, a fact from the predecessor whitepaper | answer, via tool |

- **Input:** Kokoro-synthesized speech, held in memory.
- **Rounds:** 3 per policy. Each round is one conversation, 45 turns per policy.
- **Clock:** starts at end of speech. First audio and answer start come from a serial
  playback timeline: each clip starts when it is ready and the previous clip has ended.
- **Independent baselines:** `bench_independent.py` runs each model alone in a fresh
  process, with no ASR, TTS or other model.
- **Fairness:** Ollama models were unloaded, and the script asserts it.
- **Correctness:** substring match on the reply (`contains_any`), or the current hour
  for the time question.

## 4. Results

### 4.1 Headline (warm medians, 45 turns per policy)

| Metric | Single 4B (predecessor) | A `always_filler` | B `route_aware` |
|---|---:|---:|---:|
| First audio, all turns | 860ms | 313ms | **373ms** |
| First audio, backchannel turns | n/a | 309ms | **314ms** |
| Fact answer starts playing | 860ms | 1041ms | **616ms** |
| Tool answer starts playing | n/a | 3174ms | **2899ms** |
| Routing correct | n/a | 39/45 | 39/45 |
| Facts correct | 8/8 | 21/24 | 21/24 |
| Tool answers correct | n/a | 6/9 | 6/9 |

- **Claim check:** first audio ≤ 400ms, facts ≤ 1000ms, routing ≥ 80%, speaker slowdown
  ≤ 2x.
  - A fails two checks.
  - B passes all four.
- **Correctness:** every wrong answer in both policies traces to one of two routing
  misses (§5.4).

### 4.2 Each model alone vs combined

| Model | Alone | Combined, A | Combined, B | Memory active / peak |
|---|---|---:|---:|---:|
| 270m filler decision | 18ms | 95ms (5.3x) | 95ms (5.3x) | 245 / 339MB |
| 1b speaker TTFT | 35ms (146 tok/s) | 134ms (3.9x) | 34ms (1.0x) | 717 / 767MB |
| qwen3.5 reasoner TTFT | 147ms | 145ms | 150ms | 2329 / 3175MB |
| All three loaded | | | | 3793 / 4577MB |

Accuracy of each model alone:

| Model | Result |
|---|---|
| Reasoner | Tool answers 9/9, facts 24/24 |
| Speaker | Facts 21/24; tool questions 0/9, answered by guessing |
| Filler | Routing 42/45 |

### 4.3 Where a route-aware turn spends its time

| Route | Timeline (ms after end of speech) |
|---|---|
| Backchannel | ASR 214 → filler decision +95 → bank clip plays at ~314 |
| Plain fact | ASR 214 → qwen first token ~363 (no tool, cancelled) → 1b TTFT +34 → first sentence → Kokoro → plays at ~616 |
| Tool | ASR 214 → qwen `<tool_call>` ~363 → hold clip plays at ~370 → tool round trip → 1b voices result → plays at 1.8–4.0s |

## 5. Analysis

### 5.1 One GPU turns parallel tracks into contention

- **What contends:** all tracks share the M4's GPU.
- **Policy A:** the reasoner decodes a full answer for every plain fact, while the
  speaker decodes the same answer.
  - The speaker's TTFT rises 3.9x.
  - The reasoner's output is thrown away.
- **Policy B:** cancelling the reasoner at its first token when it doesn't call a tool
  frees the GPU. The speaker returns to its solo 34ms.
- **The remaining contention:** the 270m decides while the reasoner prefills (5.3x), and
  it still decides in 95ms.
- **Rule:** start speculative work early, and cancel it the moment its result is known
  to be unused.

### 5.2 A filler is a commitment

- **A playing clip can't be cut mid-word.** Bank clips last 395–905ms even after speed
  and trim tuning.
- **Under A,** fact answers were ready at 764ms but waited behind the filler until
  1041ms. The filler made the reply later.
- **When a hold filler pays off:** only when the expected wait is well beyond its own
  length. In this workload, that is tool turns (1.8–4.0s).
- **Backchannels are different:** they are the whole reply, so they always pay off.

### 5.3 Routing belongs to the model that knows the task

| Router | Result |
|---|---|
| 1b told to reply "CHECKING" for tool questions | Never did (0/9); guessed wrong (17×23 → "one hundred eighty-nine") |
| 1b with few-shot CHECKING examples | Deferred 6/8 plain facts |
| 4B tool-calling model's first token | 9/9 tool questions, ~150ms after the question with a pinned prompt |

- **The 4B already decides tool use** as part of normal generation, so the routing signal
  costs nothing extra.
- **It over-uses tools on 2 of 8 facts:**
  - `calculate` for "days in a leap year"
  - `search_docs` for "Mount Fuji"
- **Cost of that:** both answers stay correct, but arrive in 1.6–2.0s instead of ~0.6s.

### 5.4 The 270M model is the weakest link

- **Prompt format matters:**
  - As a single-message list prompt, it answered "Okay." to 15/15 items.
  - As chat-format few-shot, it routes 42/45 alone and 39/45 in the pipeline.
- **It often answers instead of picking a phrase** ("Paris.", "Blue."). The matcher
  sends any unknown output to hold, which is the safe default.
- **Two deterministic misses, repeated every round:**
  - "who wrote Romeo and Juliet." (ASR lower case, no question mark) → "Yes."
  - "What time is it right now?" → "Okay."
- **Impact:** both become silent backchannels. They account for every wrong answer in
  §4.1.
- **Measured alternative:** the 1b routed 15/15 with clean phrases, at 184ms instead of
  41ms (unpinned, alone).

### 5.5 Prefix pinning makes fixed prompts nearly free

- **gemma3 (trimmable cache):** prefill the few-shot prefix once, then trim back after
  each call. Filler TTFT is 11ms.
- **Qwen3.5 (hybrid attention, untrimmable):** copy the prefix cache's state arrays
  once, and restore from the copy per call (~0.3ms).
  - Reasoner TTFT drops 1226ms → 147ms.
  - The generated tool call is byte-identical.

### 5.6 Implementation traps

| Trap | Effect | Fix |
|---|---|---|
| gemma tokenizer's `eos_token_ids` = `{<eos>}` only | Generation runs past `<end_of_turn>` | Add `<end_of_turn>` to stop ids |
| `detokenizer.last_segment` consumes on read | Reading it twice dropped all streamed text | Read once per step |
| Qwen3.5 template re-renders past turns without the empty think block | Cache prefix never matches | Extend the cache with the template-rendered delta only |
| Filler bank synthesized at normal settings | Clips up to 1.1s | Speed 1.15, 40ms padding |

## 6. Design rules

1. **Give each track one job:**
   - the smallest model for the fastest decision
   - a mid model for speech
   - the capable model for tools and routing
2. **Route on the capable model's first token.** Don't ask a small model to recognize
   its own limits.
3. **Cancel speculative work at the first token that proves it unused.** On one GPU,
   unused decoding is paid for by the track the user is waiting on.
4. **Play backchannels immediately.** Play hold fillers only when the predicted wait
   exceeds the clip. Never queue a filler ahead of a ready answer.
5. **Pre-synthesize the phrase bank.** Pin every fixed prompt prefix: trim where the
   cache allows, snapshot where it doesn't.
6. **Measure every model alone first.** Combined numbers without a solo baseline hide
   contention.

## 7. Limitations

- **Workload:**
  - Scripted, synthesized speech; 15 items, 3 rounds, one machine.
  - No live microphone, endpointing, playback device or barge-in.
- **Routing accuracy is for this item set only.** Real conversational speech will differ,
  and the 270m's failures look sensitive to punctuation and phrasing.
- **Naturalness:** filler naturalness and the chosen phrase's fit were not evaluated by
  listeners.
- **Shared context** carries tool results into later turns, but no multi-turn task
  exercised it.
- **Tool coverage:** `search_docs` is a keyword index over two documents, not embedding
  RAG. Reasoner thinking mode was off.
- **Memory:** the 6.5GB process RSS includes Whisper and Kokoro. Total machine memory
  with other voice agents running was not measured.

## 8. Next steps

- **Router:** replace the 270m with the 1b, or a classifier head, and measure the latency
  cost against the routing gain.
- **Filler playback:** make hold fillers cancellable at a word boundary, or predict wait
  length before committing to a clip.
- **Live test:** run the route-aware design as a LiveKit agent on live speech, next to
  single-model agents on each of the three models, and compare turn latency.
