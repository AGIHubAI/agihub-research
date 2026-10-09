#!/usr/bin/env bash
# Environment checks, then run notebook.ipynb headless, then check results.json.
# Usage: ./verify_pipeline.sh   (from any directory)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${SPIKE_PYTHON:-$HERE/.venv/bin/python}"
cd "$HERE"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok:   $*"; }

[ -x "$PY" ] || fail "interpreter not found: $PY (set SPIKE_PYTHON)"
"$PY" -c "import mlx_whisper, mlx_lm, kokoro, ollama, spacy, nbconvert, ipykernel" \
  || fail "missing packages in $PY"
ok "packages importable in $PY"

command -v espeak-ng >/dev/null || fail "espeak-ng missing (brew install espeak-ng)"
ok "espeak-ng"

if [ -f "${ACCENT_LORA_DIR:-/nonexistent}/weights.safetensors" ]; then
  ok "accent LoRA whisper at $ACCENT_LORA_DIR"
else
  echo "info: ACCENT_LORA_DIR not set, ASR uses the public whisper-small.en"
fi

grep -q tools <<<"$(ollama show qwen3.5:4b 2>/dev/null)" || fail "ollama qwen3.5:4b missing (ollama pull qwen3.5:4b)"
ok "ollama qwen3.5:4b (tools capability present)"

"$PY" -c "from huggingface_hub import snapshot_download as s; s('mlx-community/Qwen3.5-4B-4bit', local_files_only=True)" \
  >/dev/null 2>&1 || fail "mlx-community/Qwen3.5-4B-4bit not in HF cache"
ok "mlx-community/Qwen3.5-4B-4bit cached"

echo "running notebook headless (a few minutes)..."
"$PY" -m nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=1800 notebook.ipynb

"$PY" - <<'PY'
import json, sys
sys.path.insert(0, ".")
import pipeline as p
r = json.loads(p.RESULTS_PATH.read_text())
warm = [t for t in r["turns"] if t["phase"] == "warm"]
expected = len(r["meta"]["items"]) * r["meta"]["rounds"] * 2
assert len(warm) == expected, (len(warm), expected)
assert all(t["reply_text"] and t["time_to_first_audio_ms"] for t in warm), "empty reply or no audio"
for name in ("ollama", "mlx"):
    assert p.summarize(r["turns"], name) == r["summary"][name], name
v = r["verdict"]
print(json.dumps({"prewarm": r["prewarm"], "summary": r["summary"], "verdict": v}, indent=1))
print("claim holds:", v["claim_holds"], "| ttft winner:", v["ttft_winner"])
PY
ok "results.json: all warm records present, summary recomputes"
