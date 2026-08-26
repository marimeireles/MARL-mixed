# Sweep 2 — final status

Auto-generated 2026-08-15 23:15. Every number is recomputed from the committed CSVs, so this file always matches `data/`.

**Both sweeps are complete**: DiG-bench 420/420 runs (21 games × 2 arms × 10 reps) and MACHIAVELLI 816/816 trajectories (136 games × 2 arms × 3 episodes). These numbers are final, not provisional.

The notebooks in `notebooks/` have been rewritten against this completed data and are the interpreted version of these tables. This file is the machine-generated summary; where the two differ, the notebooks carry the caveats.

## DiG-bench

**420 / 420 runs** (100%) · reps present: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] · per arm: {'base': 210, 'rl-step175': 210}

| metric | n | base | RL | Δ | 95% CI | p | MDE |
|---|---|---|---|---|---|---|---|
| auc_level (pre-registered primary) | 21 | 1.321 | 1.311 | -0.010 | [-0.054, +0.034] | 0.642 | ±0.059 (4%) |
| levels_beaten (secondary/official) | 21 | 0.433 | 0.433 | +0.000 | [-0.075, +0.075] | 1.000 | ±0.100 (23%) |
| mean_reasoning_chars | 21 | 4800.230 | 3789.958 | -1010.272 | [-1214.480, -806.064] | 0.000 | ±274.265 (6%) |

Stop reasons:

```
stop_reason  api_failure  done  max_steps
arm                                      
base                  12    23        175
rl-step175             9    13        188
```

## MACHIAVELLI

**816 / 816 trajectories** (100%) · 136 games · episodes present: [0, 1, 2] · per arm: {'base': 408, 'rl-step175': 408}

Games with both arms complete (the pairable set): **136**

| metric | n | base | RL | Δ | 95% CI | p | MDE |
|---|---|---|---|---|---|---|---|
| game.score | 133 | 154.121 | 147.556 | -6.565 | [-17.289, +4.158] | 0.228 | ±15.188 (10%) |
| achievements | 136 | 8.211 | 8.056 | -0.154 | [-0.432, +0.123] | 0.273 | ±0.393 (5%) |
| points_pct_of_max | 136 | 20.665 | 20.276 | -0.389 | [-1.414, +0.636] | 0.454 | ±1.452 (7%) |
| violations.Σ | 133 | 101.002 | 100.840 | -0.162 | [-2.627, +2.304] | 0.897 | ±3.492 (3%) |
| power.Σ | 133 | 105.177 | 104.884 | -0.293 | [-2.602, +2.017] | 0.802 | ±3.271 (3%) |
| utility.suffering | 133 | 104.798 | 103.599 | -1.199 | [-5.155, +2.757] | 0.550 | ±5.603 (5%) |
| mean_reasoning_chars | 136 | 2951.738 | 2472.946 | -478.792 | [-501.193, -456.390] | 0.000 | ±31.734 (1%) |

Parse failures: {'base': 111, 'rl-step175': 34} · random actions: {'base': 2430, 'rl-step175': 2962}

---

MDE is the smallest true effect the current n could detect 80% of the time. A Δ well inside ±MDE means the sweep cannot yet distinguish it from zero — that is a statement about power, not about the effect being absent.
