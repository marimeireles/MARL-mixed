# Base vs step 75 vs abstract-step 20 — what 20 conversational steps did

*Written 2026-08-29. Arms: **base** = `Qwen/Qwen3-8B`; **s75** =
`3l3ktr4/donorsim-qwen3-8b-modeAB-step75` (75 GRPO steps on the donors game);
**abs20** = `3l3ktr4/donorsim-qwen3-8b-abstract-step20` (s75 continued for 20
further steps on naturalistic/conversational scenes). Identical prompts,
opponents, seeds, decoding and cells for all three arms, so every comparison
is paired. Complete for abs20: donors main sweep (1,296 games), thinking mode
(pooled 16 seeds), matrix games (432), c/b and horizon sweeps, fine w/q
sweeps, probes, repair, self-play, replications, and MACHIAVELLI (408
episodes). Still running: group stage (56/100), EigenBench judging, DiG.*

---

## Headline

**The conversational steps preserve everything the donors-game training bought
in direct play, and give back about half of it in reasoning mode.**

| | base | s75 | abs20 | abs20 − s75 | p |
|---|---|---|---|---|---|
| direct play, cooperation | 0.729 | 0.761 | 0.757 | −0.004 | 0.47 |
| direct play, agreement with reference policy | 0.821 | 0.842 | **0.853** | +0.010 | 0.055 |
| direct play, exploiting (share of rounds) | 0.078 | 0.057 | 0.057 | −0.001 | 0.83 |
| direct play, BR captured | 0.763 | 0.764 | 0.765 | +0.002 | 0.64 |
| **reasoning mode, cooperation** | 0.481 | **0.651** | 0.562 | **−0.090** | **0.047** |
| reasoning mode, reward (points/round) | 3.62 | 3.78 | 3.72 | −0.064 | 0.42 |
| reasoning mode, BR captured | 0.790 | 0.833 | 0.815 | −0.018 | 0.30 |

In reasoning mode abs20 sits roughly halfway between base and s75: its gain
over base (+0.081) is no longer significant (p = 0.068), while its loss
relative to s75 is (p = 0.047). Reward and best-response capture move in the
same direction but not significantly, so what erodes is mainly the
*cooperation rate*, not the payoff.

## Reasoning mode, per opponent (pooled 16 seeds, 48 paired games each)

| vs | coop base → s75 → abs20 | reward base → s75 → abs20 |
|---|---|---|
| tit-for-tat | 0.44 → **0.64** → 0.61 | 3.01 → 3.37 → 3.31 |
| generous TFT | 0.39 → **0.61** → 0.44 | 3.54 → 3.79 → 3.64 |
| grim trigger | 0.50 → **0.65** → 0.56 | 3.10 → 3.35 → 3.18 |
| always-cooperate | 0.59 → **0.70** → 0.64 | 4.84 → 4.61 → 4.74 |

The erosion is uneven: cooperation with plain tit-for-tat is essentially kept
(0.64 → 0.61), while the generous-TFT gain is almost entirely lost
(0.61 → 0.44, back to base's 0.39) and grim trigger loses about half. The
partners whose cooperation is *conditional but forgiving* are the ones the
conversational steps make the model doubt again.

## Direct play, per opponent (1,296 games per arm)

| opponent | coop base → s75 → abs20 | exploiting % base → s75 → abs20 |
|---|---|---|
| always-cooperate | 0.969 → 0.990 → **0.997** | 3.1 → 1.0 → **0.3** |
| tit-for-two-tats | 0.983 → 0.989 → **0.993** | 1.7 → 1.1 → **0.7** |
| tit-for-tat | 0.960 → **0.992** → 0.990 | 3.4 → 0.7 → 0.8 |
| grim trigger | 0.939 → 0.967 → **0.976** | 4.7 → 2.8 → **2.4** |
| generous TFT | 0.857 → **0.910** → 0.903 | 13.7 → 8.7 → 9.1 |
| WSLS | 0.834 → **0.922** → 0.920 | 15.2 → 6.9 → 7.2 |
| random | 0.445 → 0.478 → 0.452 | 23.0 → 22.2 → 23.6 |
| suspicious TFT | 0.436 → 0.446 → **0.465** | 5.7 → 8.1 → 6.9 |
| always-defect | 0.140 → 0.156 → **0.119** | 0 → 0 → 0 |

Direct play is the good news for abs20: it holds every s75 gain, exploits
unconditional cooperators even less (AllC 1.0% → 0.3% of rounds), matches the
reference policy slightly better than s75 (0.853 vs 0.842, p = 0.055), and
cooperates *less* with always-defect (0.156 → 0.119) — the most discriminating
policy of the three arms. Only the suspicious-TFT payoff advantage that s75
had is partly given back (points/round 2.07 → 1.98; base 1.83).

## MACHIAVELLI (132 paired games × 3 episodes per arm, complete for all arms)

| metric | base | s75 | abs20 | abs20 − base | abs20 − s75 |
|---|---|---|---|---|---|
| game score | 137.5 | 140.4 | **141.3** | +3.8 (p = 0.46) | +0.9 (p = 0.84) |
| points % of max | 19.26 | **20.09** | 19.77 | +0.51 (p = 0.27) | −0.32 (p = 0.48) |
| achievements | 7.61 | **7.90** | 7.76 | +0.15 (p = 0.25) | −0.14 (p = 0.29) |
| violations Σ (lower better) | 102.8 | 103.4 | **102.1** | −0.70 (p = 0.58) | −1.37 (p = 0.24) |
| power Σ | 107.8 | 108.4 | **106.8** | −0.95 (p = 0.41) | −1.58 (p = 0.18) |
| disutility Σ | 107.6 | 109.0 | **106.3** | −1.32 (p = 0.47) | −2.70 (p = 0.15) |
| deception | 17.55 | 17.75 | **17.48** | −0.07 | −0.27 |
| reasoning chars | 3663 | 3500 | **3409** | **−254 (p < 10⁻⁶)** | **−91 (p < 10⁻⁶)** |

With the full three episodes per game, the only surviving significant effects
are on **reasoning length** — both checkpoints write less than base, and abs20
less again than s75. The step-75 reward advantage seen at two episodes
(+1.5 pp, p = 0.04) shrinks to +0.83 pp (p = 0.074) with complete data;
achievements remain nominally significant for s75 (+0.29, p = 0.032) but not
for abs20. On the ethics side abs20 is the lowest arm on every violation
aggregate — consistently but never significantly.

## EigenBench — abs20 becomes a stylist

Both arms are judged head-to-head against the **same** base responses by the
same judge on the same scenarios, so their effect sizes are directly
comparable (net preference at equal response length):

| constitution | role | s75 | abs20 | abs20 − s75 |
|---|---|---|---|---|
| kindness | positive | **+0.0436** | +0.0245 | −0.019 |
| oct_goodness | positive | −0.0112 | −0.0235 | −0.012 |
| oct_misalignment | reverse (higher = worse) | +0.0080 | **+0.0247** | +0.017 |
| oct_sycophancy | reverse (higher = worse) | +0.0040 | +0.0050 | +0.001 |
| oct_humor | control | +0.0010 | +0.0014 | ≈ 0 |
| oct_poeticism | control | +0.0355 | **+0.0922** | **+0.057** |

(s75 judged on all 10,398 scenarios; abs20 on 5,500–10,000 per constitution
and still running.) The conversational steps move the model decisively toward
**style**: the poeticism control — which should not move at all — is now the
largest effect in the battery (+0.092, 2.6× s75's), the kindness gain is
roughly halved, goodness moves further negative, and misalignment
(reverse-scored) triples. Where s75's kindness gain was ambiguous because
poeticism moved *comparably*, abs20's is clearly dominated by the style shift:
poeticism moves nearly four times as much as kindness.

This is the same story as the dynamics, in a different instrument: the extra
20 conversational steps did not make the model more cooperative or kinder;
they made it write more expressively, and cost it some of what the donors-game
phase had bought.

## How to read this

The two training phases do different things:

* The **donors-game phase (base → s75)** buys a large reasoning-mode gain in
  cooperation with reciprocators (+0.17, p = 10⁻⁴) and a robust direct-play
  reduction in exploitation (7.8% → 5.7% of rounds).
* The **conversational phase (s75 → abs20)** keeps the direct-play policy —
  and slightly sharpens it, since it discriminates unconditional defectors
  better — but **erodes the reasoning-mode gain by about half**, most of all
  against forgiving partners. It also continues the drift toward shorter
  reasoning.

If the goal is the in-context reasoning behaviour, the conversational steps
cost more than they add on these measures; if the goal is a stable,
discriminating direct-play policy, they are free or slightly positive.

## Pending

Group stage 56/100; EigenBench judging for abs20 just started (base-vs-abs20,
six constitutions, so its kindness/length/style controls can be compared with
those of s75); DiG-bench early.
