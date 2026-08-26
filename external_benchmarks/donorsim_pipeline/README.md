# External benchmarks — donorSim RL adapter vs. un-RLed base

Out-of-distribution transfer evaluation of the donorSim GRPO/LoRA adapter on two
public benchmarks the adapter was **not** trained on:

| Benchmark | Measures | Scope run here |
|---|---|---|
| [**DiG-bench**](https://github.com/discos-research/dig-bench) | scientific discovery — inferring hidden game rules by experiment | 21 public games × 2 arms × 3 reps = **126 runs** |
| [**MACHIAVELLI**](https://aypan17.github.io/machiavelli/) | reward-vs-ethics tradeoff — power seeking, deception, harm | 30 test games × 2 arms × 1 episode = **60 trajectories** |

Everything here is derived from complete sweeps: 126/126 DiG-bench runs produced a
summary row, and 60/60 MACHIAVELLI trajectories completed without error.

## Headline results

**Neither benchmark separates the arms on its headline metric.** The adapter is
narrow: it did not transfer, in either direction.

| | base | RL step_175 | Δ | paired p |
|---|---|---|---|---|
| DiG-bench `levels_beaten` / run | 0.476 | 0.460 | −0.016 | 0.84 |
| MACHIAVELLI `game.score` (100 = random) | 141.1 | 131.7 | −9.4 | 0.36 |
| MACHIAVELLI `violations.Σ` (100 = random, ↓ better) | 102.4 | 97.6 | −4.8 | 0.16 |
| MACHIAVELLI `power.Σ` (↓ better) | 102.4 | 100.5 | −1.9 | 0.45 |

The one robust effect, and it replicates across **both** benchmarks:

| reasoning length | base | RL step_175 | Δ | p |
|---|---|---|---|---|
| DiG-bench (chars/turn) | 4741 | 3848 | −19% | < 0.001 |
| MACHIAVELLI (chars/decision) | 3054 | 2556 | −16% | < 0.001 |

The adapter **compresses the model's reasoning by ~16–19% at equal task performance**,
on two tasks entirely unrelated to its training distribution. On MACHIAVELLI this holds
for all 30 of 30 games individually.

All MACHIAVELLI harm axes move in the favourable direction under the adapter, and 10 of
13 violation categories do — but none of it is significant at n = 30. Consistent sign,
insignificant magnitude; it warrants replication, not a claim.

## Experimental design

The critical control: **both arms are served by one vLLM process from one set of base
weights.** The base model is registered as `qwen3-32b-base` and the LoRA is hot-swapped
in as `qwen3-32b-donorsim` over those same weights. The adapter is therefore the only
variable — not the server build, the sampling parameters, the machine, or the harness
version. See `scripts/serving/serve_vllm.sbatch`.

| | |
|---|---|
| Base | `Qwen/Qwen3-32B` |
| RL arm | `3l3ktr4/qwen3-32b-donorsim-loras` @ `step_175` (LoRA r=16, α=32, GRPO, Stage-B-only) |
| Serving | vLLM 0.27.1, TP=4, 1× GH200 node (CSCS Clariden), `--max-model-len 40960` |
| Sampling | the model's own `generation_config.json` (temp 0.6 / top_p 0.95 / top_k 20), **not overridden** — identical for both arms |
| Reasoning | ON for both arms |

**Why thinking is enabled for both arms.** The 32B `step_175` adapter is Stage-B-only,
and Stage B trains with thinking on. Evaluating both arms with reasoning enabled is the
in-distribution setting for the adapter, not a handicap.

### The Qwen3 tool-role thinking bug (DiG-bench)

Qwen3's chat template **does not open a `<think>` block when the last message in the
conversation has the `tool` role**, and `enable_thinking=True` does not override it. The
DiG-bench harness's tool protocol ends every turn on a `tool` message, so under
`--move-channel forced-tool` / `auto-tool` the model reasoned on turn 1 and then emitted
**zero** reasoning for turns 2–200 — on *both* arms.

Both arms were therefore run with `--move-channel guided-json`, where the next
observation arrives as a `user` message. The notebook asserts
`zero_reasoning_turns == 0` across all 126 runs.

Anyone comparing reasoning models on this harness should check for this first.

## Layout

```
data/
  digbench_runs.csv             126 rows — one per (arm, game, rep); the headline table
  digbench_turns.csv          24545 rows — one per turn; level traces and per-turn effort
  digbench_games.json                     game → difficulty tier (1–7) mapping
  machiavelli_trajectories.csv   60 rows — one per (arm, game, episode), with the
                                          benchmark's own random-normalised harm scores
notebooks/
  digbench_comparison.ipynb     executed, with figures
  machiavelli_comparison.ipynb  executed, with figures
  *.py                          the notebook sources as `# %%` cell scripts (diff-friendly)
scripts/
  serving/serve_vllm.sbatch     the dual-arm vLLM server
  serving/vllm.toml             container env (see the TRITON_CACHE_DIR note inside)
  digbench_orchestrate.py       resumable DiG-bench sweep driver
  machiavelli_run_arms.py       resumable MACHIAVELLI trajectory generator
  machiavelli_vllm_agent.py     MACHIAVELLI agent (reuses LMAgent's prompt verbatim)
  aggregate_*.py                raw run artifacts → the CSVs in data/
  build_notebooks.py            .py cell scripts → executed .ipynb
```

## Reproducing the figures

The CSVs in `data/` are sufficient to regenerate every number and every plot:

```bash
pip install pandas numpy scipy matplotlib nbformat nbclient ipykernel
cd notebooks && python ../scripts/build_notebooks.py
```

To regenerate the CSVs from raw run artifacts you additionally need the sweep outputs
(not distributed here — see below) and, for MACHIAVELLI, its `game_data/` and
`normalization_coeffs.json`.

## What is intentionally *not* in this repository

Raw run artifacts are excluded for licensing reasons:

- **MACHIAVELLI trajectory pickles** embed verbatim game text from Choice of Games
  titles, which is licensed and not ours to redistribute. Obtain `game_data.zip` from
  the [upstream repository](https://github.com/aypan17/machiavelli).
- **DiG-bench per-turn JSONL** embeds game observations from a benchmark whose other 49
  games are held private for secure evaluation.

`data/digbench_turns.csv` deliberately carries observation *lengths* and the agent's
single-keystroke action, never the observation text or the model's reasoning prose. Each
DiG-bench run's `session_id` and `playback` URL are retained in `digbench_runs.csv`, so
any run can be replayed move-for-move against the official API by anyone holding a
DiG-bench token.

## Caveats

- **MACHIAVELLI: 1 episode per game, not the published 25.** 25 would be ~250 GPU-hours.
  Per-arm absolute values are much noisier than the paper's and **should not be pasted
  into its baseline table**. The paired within-game comparison remains valid but is
  underpowered for small effects.
- **DiG-bench: 90% of runs end at the 200-turn ceiling**, not by finishing. The ceiling
  is identical for both arms so the comparison is fair, but the absolute scores are a
  lower bound, not an asymptote, and are not comparable to leaderboard entries run at a
  different step budget.
- **DiG-bench: 3 reps/game** bounds the resolvable effect size — enough to rule out a
  large effect, not a subtle one.
- **Chat template.** Serving used the stock upstream Qwen3 template; the adapter was
  trained under a patched variant (defaults `enable_thinking=True`, honours `/think`
  and `/no_think`). Behaviour matches here — upstream defaults thinking on and no
  markers are sent — but the templates are not byte-identical.
- MACHIAVELLI's harm annotations are 2023 model- and human-generated scene labels, not
  a ground-truth measure of agent intent.
