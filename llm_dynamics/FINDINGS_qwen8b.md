# Full review — Qwen3-8B base vs donorSim GRPO step 6 vs step 51

*Written 2026-08-28 from the artifacts in `llm_dynamics/results/` and
`external_benchmarks/`. Status: complete except the group-stage evaluations
(s6 78/100, s51 28/100 games), the s6-arm MACHIAVELLI tail (392/408 per arm)
and DiG-bench (~half done, all arms). Those remainders are running; nothing
below is expected to flip, but group-stage CFE/Brier and DiG are provisional.*

**Arms.** base = `Qwen/Qwen3-8B`; s6 / s51 = donorSim mode-A/B GRPO checkpoints
at steps 6 and 51 (`3l3ktr4/donorsim-qwen3-8b-modeAB-step{6,51}`). All dyadic
games use the training-faithful prompts; "thinking on" is the training mode.

---

## The headline: training moves the model in two different directions, at two different times

The most interesting single result of the whole battery is that **step 6 and
step 51 diverge from the base model in opposite directions**, and the effect
lives almost entirely in **thinking mode** — the mode the model was trained in.

**Donors game, thinking ON (w=1), cooperation with cooperators:**

| vs | base | s6 | s51 |
|---|---|---|---|
| tit-for-tat | 0.36 | 0.35 | **0.68** |
| grim trigger | 0.42 | 0.34 | **0.71** |
| always-cooperate | 0.47 | 0.42 | **0.72** |
| generous TFT | 0.19 | **0.62** | 0.43 |
| overall (all 9 strategies) | 0.33 | 0.35 | **0.44** |
| BR captured overall | 0.79 | 0.80 | **0.82** |

The base model in thinking mode is a heavy defector even against perfect
reciprocators (0.36 vs TFT — it reasons its way into exploitation and pays for
it: BR captured only 0.70 vs TFT). Step 6 nudged this *further down* against
TFT/Grim/AllC. By step 51 the direction has reversed decisively: cooperation
with every reciprocator roughly doubles, and — crucially — this is not
unconditional niceness, because **payoff capture rises with it** (BR captured
vs TFT 0.70 → 0.84, vs Grim 0.72 → 0.85) and cooperation with always-defect
stays at 0.02 with ρ = +1.00. Step 51 in its training mode is a *better
reciprocator*: nicer where niceness pays, unmoved where it doesn't. The ρ
(Term 2) signature confirms it: vs AllC the base's ρ is −0.08 (it keeps
breaking coordination) while s51's is +0.42.

This matches the paper's "Future work" observation of a **non-monotonic
training trajectory** — harmful/exploitative movement early (s6), cooperative
later (s51) — and it is now quantified across nine opponents.

**Without thinking, all three models are nearly the same policy** (overall
cooperation 0.745 / 0.722 / 0.752; agreement with the reference policy 0.836 /
0.821 / 0.851). The only reliable no-think divergence is s6's defection drift
against forgiving reciprocators — IPD vs TF2T: coop 1.00 → 0.75 (points 3.00 →
2.83); vs Grim 0.69 → 0.54 (2.40 → 2.11) — which **s51 fully reverses** (TF2T
1.00, Grim 0.99, TFT 1.00 in the donors game). The games×models figure
(section 1.5) shows this: the s6 bars dip below base against TF2T on the IPD
and the s51 bars come back up.

## EigenBench (Gemma-4-31B judge, 10,398 scenarios × both orders × 6 constitutions)

Step 6 was a clean null with a small wrong-way lean (kindness 50.0%,
misalignment 52.0% "RL", everything gone at equal length). **Step 51 is not a
null** — the judge now prefers it broadly:

| constitution | role | s51 win rate | net pref | net at equal length |
|---|---|---|---|---|
| kindness | positive | **0.553 ± 0.004** | +0.064 | **+0.039 [+0.027, +0.052]** |
| oct_goodness | positive | 0.484 ± 0.005 | −0.009 | −0.017 |
| oct_misalignment | reverse | 0.538 ± 0.008 | +0.011 | +0.009 |
| oct_sycophancy | reverse | 0.586 ± 0.017 | +0.005 | +0.005 |
| oct_poeticism | control | **0.559 ± 0.006** | +0.033 | **+0.034** |
| oct_humor | control | 0.532 ± 0.027 | +0.001 | +0.000 |

Read with the battery's own discriminant logic, this is a nuanced result:

* The **kindness gain is real beyond length** (s51 is only 3% longer; the
  equal-length net is +0.039, CI excluding 0) — unlike the 32B run and unlike
  s6, where kindness effects were pure verbosity.
* But the **poeticism control moves just as much** (+0.034 at equal length),
  and sycophancy/misalignment (reverse-scored) also tick toward s51. The
  pattern "judge prefers s51 on nearly every axis, including ones that should
  not move" is the signature of a **stylistic shift the judge likes** —
  warmer, more expressive prose — rather than a specifically *moral* one.
  Goodness (15 criteria, the strictest positive constitution) does not move.
* Conclusion: s51 writes in a way judges prefer, and some of that reads as
  kindness; claim "kinder" only alongside the poeticism control, or with a
  style-controlled judge.

## MACHIAVELLI (132 paired games × 3 episodes per arm)

The step-6 reward tilt is now significant with full data, and step 51 gives it
back:

| metric (paired by game) | s6 − base | s51 − base |
|---|---|---|
| points % of max | **+1.08 (p = 0.026)** | +0.03 (n.s.) |
| achievements | **+0.29 (p = 0.034)** | +0.14 (n.s.) |
| violations.Σ (normalized) | −0.4 (n.s.) | −1.9 (n.s., 71/132 games *lower*) |
| power.Σ / utility.Σ | ≈ 0 | −1.8 / −0.5 (n.s.) |
| reasoning chars | **−29 (p < 0.001)** | **−204 (p < 0.001, shorter in 131/132 games)** |

So: s6 = slightly better at the games, ethics unchanged (the early
stealing/vandalism signal from the 45-game interim washed out with full data —
a good lesson in waiting). s51 = back to base reward, violations pointing
mildly *down* across most games without significance, and a large, extremely
consistent drop in reasoning length — the same "RL shortens reasoning" effect
the 32B runs showed (4800→3790 chars there), now 3662→3458 here.

## DiG-bench (partial: 102/105/68 runs of 210 per arm)

Nothing separates the arms: auc_level 1.19 / 1.20 / 1.16, paired-by-game
differences ~0 (p = 0.72 / 0.37). Discovery ability is untouched by this
training — neither harmed nor helped. (Consistent with the 32B result:
auc 1.32 vs 1.31, p = 0.64.)

## Dynamics odds and ends

* **Repair after a forced defection** (both arms have this): unchanged by
  training — the no-think policies are near-identical, so forgiveness
  (P(C|cd) ≈ 0) stays the base model's missing skill in every arm.
* **Self-play**: 100% mutual cooperation for all arms at every (w, q) tested.
* **c/b sweep, horizons, memory windows**: no systematic differences beyond
  seed noise; the note-style compaction memory behaves like the full window.
* **Term-2 saturation held throughout**: mean ρ is unchanged in no-think play
  for all arms (~+0.43 overall); the training signal that *did* move behaviour
  evidently came from the thinking-mode group-stage rounds (Terms 1/3/bonus),
  not from the reciprocation term — exactly what the saturation analysis
  predicted before the first RL checkpoint was evaluated.

## Where the models diverge the most — ranked

1. **Thinking-mode cooperation with reciprocators** (donors game, w=1): base
   0.36–0.47 → s51 0.68–0.72, with payoff capture *up*. Largest, most
   coherent effect in the battery; opposite sign at s6.
2. **Judge-perceived style/kindness** (EigenBench): s51 +5.3% kindness win
   rate, surviving the length control but mirrored by the poeticism control.
3. **Reasoning length** (MACHIAVELLI): s51 writes ~6% less reasoning in 131 of
   132 games — the most statistically extreme effect anywhere (p < 10⁻⁶),
   though behaviourally the smallest.
4. **s6's transient exploitation of forgiving partners** (IPD vs TF2T/Grim),
   erased by s51.
5. **Everything else is a null**: no-think donors dyads, matrix games at s51,
   DiG, MACHIAVELLI ethics scores, repair, self-play.

**One-sentence summary:** between step 6 and step 51 the GRPO run turns from a
slight exploitation drift into a genuine gain in *conditional* cooperation —
visible only in the training (thinking) mode, accompanied by a style shift
judges read as kindness and by consistently shorter reasoning, with game-solving
ability and ethics-violation profiles unchanged.
