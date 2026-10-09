#!/usr/bin/env bash
# Checks, independent per-model benchmarks (fresh process each), combined notebook run.
# Usage: ./verify_multitrack.sh [--skip-independent]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${SPIKE_PYTHON:-$HERE/.venv/bin/python}"
cd "$HERE"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok:   $*"; }

[ -x "$PY" ] || fail "interpreter not found: $PY (set SPIKE_PYTHON)"
"$PY" -c "import mlx_whisper, mlx_lm, kokoro, ollama, spacy, nbconvert, ipykernel" \
  || fail "missing packages in $PY"
ok "packages importable"

for repo in mlx-community/gemma-3-270m-it-qat-4bit mlx-community/gemma-3-1b-it-qat-4bit \
            mlx-community/Qwen3.5-4B-4bit; do
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$repo', local_files_only=True)" \
    >/dev/null 2>&1 || fail "$repo not in HF cache"
  ok "$repo cached"
done

# Fairness: nothing may sit in Ollama while mlx is measured.
"$PY" - <<'PY'
import ollama, time
for m in ollama.ps().models:
    ollama.generate(model=m.model, prompt="", keep_alive=0)
time.sleep(2)
assert not ollama.ps().models, "ollama models still loaded"
PY
ok "ollama has no models loaded"

if [ "${1:-}" != "--skip-independent" ]; then
  rm -f results_independent.json
  for track in filler speaker reasoner; do
    echo "independent: $track"
    "$PY" bench_independent.py --track "$track" --repeat 3 >/dev/null 2>&1 \
      || fail "bench_independent $track"
  done
  ok "results_independent.json written"
fi

for policy in always_filler route_aware; do
  echo "running combined notebook headless, policy=$policy..."
  cp notebook.ipynb "notebook_$policy.run.ipynb"
  MULTITRACK_POLICY=$policy "$PY" -m nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=3600 "notebook_$policy.run.ipynb"
done
# the committed notebook keeps the outputs of the default policy
mv notebook_route_aware.run.ipynb notebook.ipynb
mv notebook_always_filler.run.ipynb notebook_always_filler.ipynb

"$PY" - <<'PY'
import json
import tracks as T
for policy in ("always_filler", "route_aware"):
    r = json.loads((T.HERE / f"results_combined_{policy}.json").read_text())
    assert len(r["turns"]) == r["meta"]["rounds"] * len(r["meta"]["items"])
    assert T.summarize(r["turns"]) == r["summary"]
    print(policy, json.dumps({"summary": r["summary"], "contention": r["contention"],
                              "verdict": r["verdict"]}, indent=1))
PY
ok "results_combined_*.json consistent"
