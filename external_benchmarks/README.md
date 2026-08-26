# External benchmarks (DiG-bench, MACHIAVELLI, EigenBench)

Reuses the pipeline built for the 32B arms in donorSim@neurips-methodology
(`donorsim_pipeline/` = verbatim copies of its scripts, data and READMEs),
adapted to run against an `llm_dynamics/slurm/serve_model.sbatch` vLLM server on
this cluster. `./setup.sh` reproduces everything (clones, data, venvs, patches).

| benchmark | what it measures | runner | job script |
|---|---|---|---|
| MACHIAVELLI (136 working games × N episodes) | reward vs. ethical violations / power / utility, normalized to a random agent (100 = as harmful as random) | `machiavelli/run_arms.py` (+ `VLLMAgent`), `aggregate_machiavelli.py` | `slurm/bench_machiavelli.sbatch` |
| DiG-bench (21 public games × reps) | scientific discovery by experimentation; `auc_level`, `levels_beaten` | `dig-bench/orchestrate.py` (baseline harness, guided-json move channel), `aggregate_digbench.py` | `slurm/bench_digbench.sbatch` — needs `DIGBENCH_API_TOKEN` |
| EigenBench (AIRiskDilemmas, 10 398 scenarios) | value alignment vs. a constitution; **responses only here**, judging done externally (donorSim design: gpt-oss-120b judge, both orders, `<think>` stripped) | `EigenBench/runs/qwen8b/gen_responses.py` | `slurm/bench_eigen_responses.sbatch` (GPU) |

The server must be launched with `EXTRA_ARGS="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes"` and `MAX_LEN=40960` (DiG needs the window; MACHIAVELLI's agent reads `reasoning_content`).
Arms are `{"label": "served-model-name"}` maps (`MACH_ARMS`, `DIG_ARMS`); add the trained model as a second served name (LoRA via `LORA=` in `serve_model.sbatch`) to get a paired comparison.
Outputs land in `runs/` (gitignored; aggregate CSVs are copied to `../llm_dynamics/results/external/` when a run completes).
