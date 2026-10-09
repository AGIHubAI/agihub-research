# AGIHub Research

Measurements and methodology for local, low-latency voice agents on consumer hardware.

## Layout

Monorepo of independent experiments. One folder per experiment at the root, named
`YYMMDD-<topic>` by start date.

- Each folder is self-contained: own code, data, `requirements.txt` and README.
- No code is shared between folders; what an experiment needs is copied into it.
- Each README states the claim, the falsifier, the outcome and how to run it.

## Experiments

| Folder | Claim | Outcome |
|---|---|---|
| [`260917-voice-pipeline-qwen35`][e1] | qwen3.5:4b voice turn, first audio < 1.5 s | mlx-lm yes, Ollama no |

[e1]: 260917-voice-pipeline-qwen35/

## Run an experiment

```bash
git clone https://github.com/AGIHubAI/agihub-research.git
cd agihub-research/<experiment>
# then follow that folder's README: venv, requirements, run script
```

## Scope

| Area | What it covers |
|---|---|
| Voice latency | Time from end of speech to first audio; streaming ASR -> LLM -> TTS |
| ASR and accents | Whisper on Apple silicon, accent LoRA, evaluation sets |
| Multi-track voice | Fast filler model plus slower reasoner in one turn |
| Methodology | Hypothesis-driven development, AI software factory, agent teams |

## Conventions

- Claim and falsifier are written before measuring.
- Numbers come from runs on named hardware; synthetic test sets are labelled as such.
- Papers are referenced by arXiv id and link; full texts are not redistributed.

## License

Apache-2.0. See `LICENSE`.
