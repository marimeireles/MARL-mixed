# EigenBench: value alignment of the donorSim adapter

Third external benchmark for the step_175 adapter, alongside DiG-bench (discovery)
and MACHIAVELLI (reward-vs-ethics). **Not yet run** — this directory is the
prepared configuration, committed ahead of the scratch purge.

Upstream: https://github.com/jchang153/EigenBench

## What EigenBench does differently

There is no ground truth and no external arbiter. Every model in the run judges
every other model's free-text responses against a **constitution** (a list of
pairwise preference criteria); a Bradley-Terry-Davison model fits relative
scores from those comparisons, and EigenTrust re-weights each judge by how well
it agrees with consensus.

Two consequences that shape the design below:

1. **You cannot run two models.** With only our two arms the BTD fit collapses
   to a single Elo difference and a 2×2 trust matrix has no consensus to
   extract. External models must be in the panel for the arms to be ranked
   against.
2. **Judge families must be diverse.** EigenTrust reads inter-judge agreement as
   reliability, so two models from one lab become a voting bloc that dominates
   the consensus.

## Panel — 4 models, 3 families

| model | role | size |
|---|---|---|
| `Qwen/Qwen3-32B` | control arm | (already needed) |
| `3l3ktr4/qwen3-32b-donorsim-loras` @ `step_175` | treatment arm | LoRA over the same weights |
| `google/gemma-4-26B-A4B-it` | external judge | 52 GB, MoE 4B active |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | external judge | 48 GB |

Both arms share one set of base weights with the adapter hot-swapped, the same
control used in the other two benchmarks. EigenBench supports this natively
(`enable_lora` + per-model `lora_request`), and its `base_model_id` field
overrides the adapter's stale `/dev/shm/verl-cache/...` base path, so the manual
`adapter_config.json` patch is not needed here.

**Known bias to report around:** Qwen3-32B judges its own LoRA descendant.
Per-judge comparisons are all kept in `evaluations.jsonl`, so refit the BTD with
Qwen excluded as a judge and confirm the ranking survives — no re-run required.

## Scenarios

`data/airiskdilemmas.json` — 400 dilemmas from
[AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas),
deduplicated from 10,398 unique dilemmas and stratified across its 9 risk
categories (~46 each: Deception, Power-Seeking, Self-Preservation, Proxy Gaming,
Corrigibility Failures, Privacy Violation, Alignment Faking, Others,
unlabelled). Regenerate with `build_scenarios.py`.

Chosen over the paper's other two datasets (AskReddit, OpenAssistant) because
behaving well has to *cost* something for the measurement to have dynamic range.
A kindness rubric applied to open chit-chat scores both arms alike and measures
noise — the same floor effect that made DiG-bench's `levels_beaten` a poor
endpoint.

Each entry keeps the dataset's `risky_behaviors` and `context` labels alongside
the `dilemma` text. EigenBench reads only `dilemma` and ignores the rest, so
they cost nothing at run time but allow slicing results by risk category
afterwards.

## Constitution battery

One run per constitution, same scenarios and same panel throughout. The battery
is what makes any result interpretable:

| role | constitutions | expectation if the adapter genuinely helps |
|---|---|---|
| positive | `kindness` (8), `oct_goodness` (15) | RL arm **higher** |
| reverse-scored | `oct_misalignment` (10), `oct_sycophancy` (10) | RL arm **lower** |
| discriminant control | `oct_humor` (10), `oct_poeticism` (10) | **no movement** |

The reverse-scored constitutions reward taking pleasure in harm and flattery
respectively, so a well-behaved model should score *low*. The discriminant row
is the control our MACHIAVELLI analysis lacked: if the RL arm rises on every
axis including humour and poeticism, that is judges liking it more, not ethics.

## Running it

```bash
git clone https://github.com/jchang153/EigenBench.git
cd EigenBench && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

mkdir -p runs/donorsim data/scenarios
cp <this dir>/{spec.py,build_scenarios.py,run_battery.sh} runs/donorsim/
cp <this dir>/data/airiskdilemmas.json data/scenarios/

python runs/donorsim/fetch_judges.py       # ~100 GB, external judges only
bash runs/donorsim/run_battery.sh          # full battery
# or a single constitution:
python scripts/run.py runs/donorsim/spec.py
```

No API key is required — the panel is entirely local via vLLM. `OPENROUTER_API_KEY`
is only needed if frontier models are added to the panel.

## Cost

`sampler_mode: all_to_all` — with 4 models this is affordable and strictly better
than sampling, since every arm is scored by every judge on every scenario and no
arm gets a lucky judge draw. Per constitution: 1,600 response generations
(400 × 4 models) and 1,600 judging calls (400 × 4 judges). Responses are
generated once and cached across the battery by `run_battery.sh`, so only the
first constitution pays for generation.

## Status

Configured and validated against EigenBench's own loaders (spec parses, LoRA
reference resolves, 400 scenarios load, criteria counts match each constitution
file). All four models are on disk as of 2026-08-15 — the two Qwen arms from
RECOVERY.md plus both judges via `fetch_judges.py`. Not executed: waiting on the
DiG-bench sweep to release the node.

Note on `fetch_judges.py`: it verifies completion by measuring safetensors bytes
on disk, not by exit code. The `hf download` CLI in huggingface_hub 1.x reads
trailing patterns after `--exclude` as explicit *filenames*, so the obvious
shell version downloads only config files and still exits 0.

## Withdrawn result (2026-08-15)

A 6-scenario smoke run completed and produced Elo ratings — step_175 1546.15,
base 1535.78, Gemma 1450.26, Mistral 1441.23. **Those numbers are withdrawn.
Do not cite them, including as a weak or directional result.**

They were generated before `_visible()` existed in
`pipeline/eval/mixed_collect.py`. Until then nothing stripped Qwen3's `<think>`
scratchpad from the text the judges read, so every comparison set "scratchpad +
answer" against "answer alone":

| model | seen by judge | of which `<think>` |
|---|---|---|
| Qwen3-32B base | 5,536 chars | 3,272 |
| donorSim step_175 | 4,896 | 3,087 |
| Gemma 3 27B | 1,562 | 0 |
| Mistral Small 3.2 24B | 1,181 | 0 |

That is a 4.7x cross-family length asymmetry caused by prompt-template handling
rather than by content, and judges prefer longer responses. It also falls
directly on the arm comparison: the two arms differ in *visible* answer length
by −20.3%, because the RL adapter reasons less — the same effect measured on
MACHIAVELLI (−16.2%) and DiG (−21.0%). An arm difference measured this way is
confounded with that effort reduction, so it could have been reported as a
kindness result when it was partly a verbosity result.

The BTD fit was deleted rather than archived, so the figures cannot be quoted
from disk. The raw judged records are kept outside the repo purely as evidence
of the defect. Nothing downstream ever consumed these numbers — they were
reported in conversation only, and no committed file referenced them.

Every score after this point comes from the patched pipeline (`<think>` stripped
from all judge input, TP=4). Results from before and after the patch are not
comparable and must not be pooled.
