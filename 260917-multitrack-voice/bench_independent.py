#!/usr/bin/env python3
"""
Measure one track's mlx 4-bit model alone, in a fresh process: no other model loaded,
no ASR and no TTS. The notebook's combined run is compared against these numbers.

Usage (one process per track, so memory figures are per model):
    python bench_independent.py --track filler   --repeat 3
    python bench_independent.py --track speaker  --repeat 3
    python bench_independent.py --track reasoner --repeat 3
Appends to results_independent.json (replacing an earlier record for the same track).
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import median

import tracks as T

OUT = T.HERE / "results_independent.json"


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(median(vals), 1) if vals else None


def spread(vals):
    vals = [v for v in vals if v is not None]
    return round(max(vals) - min(vals), 1) if vals else None


def bench_filler(repeat: int, items: list[dict]) -> dict:
    track = T.FillerTrack()
    load_ms = track.load()
    t = time.monotonic()
    track.pick("Warm up call.")
    first_ms = T.ms(t)
    calls = [track.pick(i["text"]) | {"id": i["id"], "expected": i["route"]}
             for _ in range(repeat) for i in items]
    return {
        "repo": T.FILLER_REPO, "load_ms": load_ms, "first_call_ms": first_ms,
        "prefix_pinned": track.trimmable, "calls": len(calls),
        "ttft_ms": med(c["filler_ttft_ms"] for c in calls),
        "decision_ms": med(c["filler_decision_ms"] for c in calls),
        "decision_ms_spread": spread(c["filler_decision_ms"] for c in calls),
        "route_accuracy": f"{sum(c['route'] == c['expected'] for c in calls)}/{len(calls)}",
        "raw_outputs": sorted({c["filler_raw"] for c in calls}),
        "lm": track.lm,
    }


def bench_speaker(repeat: int, items: list[dict]) -> dict:
    track = T.SpeakerTrack(T.SharedContext())
    load_ms = track.load()  # includes one warm turn
    calls = []
    hold = [i for i in items if i["route"] == "hold"]
    for _ in range(repeat):
        track.new_conversation()
        for item in hold:
            track.history.append({"role": "user", "content": item["text"]})
            text = track.lm.chat(track._messages()).strip()
            track.history.append({"role": "assistant", "content": text})
            calls.append(dict(track.lm.last, id=item["id"], text=text,
                              correct=T.is_correct(item, text)))
    facts = [c for c in calls if c["id"].startswith("fact-")]
    tools = [c for c in calls if c["id"].startswith("tool-")]
    return {
        "repo": T.SPEAKER_REPO, "load_and_warm_ms": load_ms, "calls": len(calls),
        "ttft_ms": med(c["ttft_ms"] for c in calls),
        "ttft_ms_spread": spread(c["ttft_ms"] for c in calls if c["cache_hit_tokens"]),
        "ttft_ms_turn2plus": med(c["ttft_ms"] for c in calls if c["cache_hit_tokens"]),
        "total_ms": med(c["total_ms"] for c in calls),
        "decode_tok_s": med(c["decode_tok_s"] for c in calls),
        "facts_correct": f"{sum(bool(c['correct']) for c in facts)}/{len(facts)}",
        "tools_correct_without_tools": f"{sum(bool(c['correct']) for c in tools)}/{len(tools)}",
        "sample": {c["id"]: c["text"] for c in calls[: len(hold)]},
        "lm": track.lm,
    }


def bench_reasoner(repeat: int, items: list[dict]) -> dict:
    track = T.ReasonerTrack(T.SharedContext())
    load_ms = track.load()  # includes pinning the tool-prompt prefix
    t = time.monotonic()
    track.answer("What is two plus two?")
    first_ms = T.ms(t)
    work = [i for i in items if i["route"] == "hold"]
    calls = []
    for _ in range(repeat):
        for item in work:
            r = track.answer(item["text"])
            calls.append(dict(r, id=item["id"], correct=T.is_correct(item, r["answer"])))
    tools = [c for c in calls if c["id"].startswith("tool-")]
    facts = [c for c in calls if c["id"].startswith("fact-")]
    tool_of = {i["id"]: i["tool"] for i in T.EXTRA_ITEMS if "tool" in i}
    return {
        "repo": T.REASONER_REPO, "load_and_pin_ms": load_ms, "first_call_ms": first_ms,
        "calls": len(calls),
        "ttft_ms": med(c["reasoner_ttft_ms"] for c in calls),
        "ttft_ms_spread": spread(c["reasoner_ttft_ms"] for c in calls),
        "total_ms_tools": med(c["reasoner_total_ms"] for c in tools),
        "total_ms_facts": med(c["reasoner_total_ms"] for c in facts),
        "facts_tool_routed": f"{sum(bool(c['reasoner_tool']) for c in facts)}/{len(facts)}",
        "tools_tool_routed": f"{sum(bool(c['reasoner_tool']) for c in tools)}/{len(tools)}",
        "tools_expected_tool_used": f"{sum(tool_of[c['id']] in [t['name'] for t in c['tools']] for c in tools)}/{len(tools)}",
        "tools_correct": f"{sum(bool(c['correct']) for c in tools)}/{len(tools)}",
        "facts_correct": f"{sum(bool(c['correct']) for c in facts)}/{len(facts)}",
        "sample": {c["id"]: c["answer"] for c in calls[: len(work)]},
        "lm": track.lm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["filler", "speaker", "reasoner"], required=True)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import mlx.core as mx

    items = T.load_items()
    record = {"filler": bench_filler, "speaker": bench_speaker,
              "reasoner": bench_reasoner}[args.track](args.repeat, items)
    record.update(track=args.track, repeat=args.repeat,
                  mlx_active_mb=round(mx.get_active_memory() / 2**20),  # model still referenced
                  mlx_peak_mb=round(mx.get_peak_memory() / 2**20))
    record.pop("lm")

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[args.track] = record
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
