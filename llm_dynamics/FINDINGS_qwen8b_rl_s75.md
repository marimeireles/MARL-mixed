# Qwen3-8B base vs donorSim GRPO step 75 — preliminary review

*Written 2026-08-28 from the artifacts in `llm_dynamics/results/` and
`external_benchmarks/`. Comparisons in this file are **base vs step 75 only**.
Status: the donors direct-play main sweep, all c/b and horizon side sweeps,
the matrix games, probes, perturbation and self-play are complete for step 75;
the thinking-mode sweep and the group stage are still running. EigenBench is ~80% judged on kindness/misalignment/humor and has just started
on goodness/sycophancy/poeticism (so no length control yet); MACHIAVELLI is in
its second episode per game (~590/1224 files); DiG-bench is ~30%. Everything
external is provisional.*

**Arms.** base = `Qwen/Qwen3-8B`; s75 = `3l3ktr4/donorsim-qwen3-8b-modeAB-step75`
(merged weights, step 75 of the same GRPO run as steps 6 and 51). Identical
prompts, opponents, seeds, decoding and cells for both arms.

---

## Headline

In direct play, step 75 is a **slightly but very consistently more cooperative
reciprocator** than the base model: +3.2 points of cooperation paired over
1,296 games (p < 0.001), a better match to the reference policy (+2.1 points,
p = 0.002), **identical best-response capture** (0.764 vs 0.763, p = 0.99),
and a small price in points (−0.02/round, p = 0.07) paid by exploiting
forgiving partners less. Cooperation with unconditional defectors does not
move. The one opponent against which step 75 earns clearly *more* is the
hostile opener (suspicious TFT), and it does so at constant cooperation — by
timing, not by volume.

## Direct play — main sweep (324 cells × 4 seeds per arm)

| | base | s75 | paired diff | p |
|---|---|---|---|---|
| cooperation rate | 0.729 | **0.761** | +0.032 | < 0.001 |
| agreement with reference policy | 0.821 | **0.842** | +0.021 | 0.002 |
| BR captured | 0.763 | 0.764 | +0.000 | 0.99 |
| welfare captured | 0.907 | **0.918** | +0.011 | — |
| points / round | 3.447 | 3.425 | −0.022 | 0.066 |

### By opponent — cooperation | points/round | BR captured

| opponent | coop base → s75 | points base → s75 | BR base → s75 | optimum pts |
|---|---|---|---|---|
| always-cooperate | 0.97 → **0.99** | 4.06 → 4.02 | 0.68 → 0.67 | 6.0 |
| tit-for-two-tats | 0.98 → **0.99** | 4.03 → 4.02 | 0.72 → 0.72 | 5.1 |
| generous TFT | 0.86 → **0.91** | 4.24 → 4.15 | 0.84 → 0.83 | 4.1 |
| tit-for-tat | 0.96 → **0.99** | 4.05 → 4.01 | 0.81 → 0.81 | 4.1 |
| grim trigger | 0.94 → **0.97** | 4.05 → 4.04 | 0.81 → 0.81 | 4.1 |
| WSLS | 0.83 → **0.92** | 4.25 → 4.10 | 0.85 → 0.82 | 4.1 |
| random | 0.45 → 0.48 | 2.78 → 2.72 | 0.70 → 0.68 | 4.0 |
| suspicious TFT | 0.44 → 0.45 | 1.83 → **2.07** | 0.61 → **0.70** | 3.9 |
| always-defect | 0.14 → 0.16 | 1.72 → 1.69 | 0.86 → 0.84 | 2.0 |

Three movements: (i) cooperation with every reciprocator rises to the ceiling
(TFT/TF2T/AllC 0.99; WSLS +0.09, the largest); (ii) cooperation with the
unconditional partners — always-defect, random — and with the hostile opener
does not change (all p > 0.4, cells split evenly); (iii) payoff moves only
against suspicious TFT (+0.24/round, BR captured 0.61 → 0.70).

### Outcome structure — suckered vs exploiting (share of rounds, %)

| opponent | suckered base → s75 | exploiting base → s75 | mutual C base → s75 |
|---|---|---|---|
| tit-for-tat | 0.1 → 0.0 | 3.4 → **0.7** | 95.9 → 99.2 |
| always-cooperate | 0.0 → 0.0 | 3.1 → **1.0** | 96.9 → 99.0 |
| tit-for-two-tats | 0.0 → 0.0 | 1.7 → **1.1** | 98.3 → 98.9 |
| grim trigger | 0.3 → 0.1 | 4.7 → **2.8** | 93.6 → 96.6 |
| generous TFT | 0.4 → 0.4 | 13.7 → **8.7** | 85.2 → 90.7 |
| WSLS | 0.7 → 0.6 | 15.2 → **6.9** | 82.7 → 91.6 |
| random | 25.7 → 28.2 | 23.0 → 22.2 | 18.9 → 19.6 |
| suspicious TFT | 31.7 → **28.5** | 5.7 → 8.1 | 11.9 → 16.0 |
| always-defect | 14.0 → 15.6 | 0 → 0 | 0 → 0 |
| **all** | 8.1 → 8.2 | 7.8 → **5.7** | 64.8 → 67.9 |

Being suckered is unchanged overall (8%) — it happens almost only against the
three partners who defect regardless, and the model always tries cooperation
first (14–16% suckered vs always-defect at both arms). What changes is
**exploiting**: step 75 defects on a cooperating partner in 5.7% of rounds vs
7.8%, and against every reciprocator the exploiting share halves or better
(WSLS 15 → 7, generous TFT 14 → 9, TFT 3.4 → 0.7). Against suspicious TFT the
picture inverts — fewer suckered rounds (32 → 28.5), more exploiting rounds
(5.7 → 8.1), more mutual cooperation (12 → 16): the same amount of
cooperation, better timed, which is where the +0.24 points come from.

### By (w, q) — cooperation rate (all opponents, memories, seeds pooled)

| w | q | base | s75 | Δ |
|---|---|---|---|---|
| 0 | 0 | 0.740 | 0.758 | +0.018 |
| 0 | 0.5 | 0.718 | 0.727 | +0.009 |
| 0 | 1 | 0.686 | 0.717 | +0.031 |
| 0.5 | 0 | 0.701 | **0.784** | **+0.083** |
| 0.5 | 0.5 | 0.742 | 0.776 | +0.034 |
| 0.5 | 1 | 0.713 | 0.723 | +0.010 |
| 1 | 0 | 0.740 | **0.790** | +0.050 |
| 1 | 0.5 | 0.767 | 0.787 | +0.020 |
| 1 | 1 | 0.757 | 0.788 | +0.031 |

The base model is nearly flat across the grid (0.69–0.77): it plays
tit-for-tat with whoever is in front of it regardless of w and q. Step 75 adds
cooperation mainly in the w ≥ 0.5 rows — a weak version of the w-gradient
that direct reciprocity predicts (w = 0: ~0.73; w = 1: ~0.79) that was not
there before. Neither arm responds to q. BR captured by (w, q) is identical
for both arms (0.66–0.70 at w = 0, 0.85–0.87 at w = 1).

### By memory window — cooperation | agreement

| memory | base | s75 |
|---|---|---|
| full | 0.745 / 0.836 | 0.746 / 0.847 |
| m1 | 0.719 / 0.792 | **0.778 / 0.836** |
| m2 | 0.728 / 0.839 | **0.767** / 0.847 |
| note2 | 0.726 / 0.818 | 0.753 / 0.840 |

The gain is largest at **memory 1** (+0.06 cooperation, +0.04 agreement) and
essentially absent at full memory: what training fixes most is the
short-window echo-cycle trap (mirroring a single defection back and forth)
that the base model falls into.

### c/b sweep (memory full; w ∈ {0.5, 1}, q ∈ {0, 1}) — cooperation | BR captured

| c/b | base | s75 |
|---|---|---|
| 0.25 (cheap) | 0.712 / 0.819 | **0.779** / 0.834 |
| 0.5 (main) | 0.750 / 0.799 | 0.763 / 0.797 |
| 0.75 (costly) | 0.729 / 0.753 | **0.760** / 0.762 |

The base model is blind to c/b (0.71 / 0.75 / 0.73 — no gradient). Step 75
cooperates more at both the cheap and the costly end without becoming
sensitive to cost either; the biggest gain is where cooperating is nearly free
(c/b = 0.25, +0.07). The two settings where cooperation should collapse are
below.

### c/b where cooperation should collapse (c/b = 1.0, 1.25) — the counterweight

| c/b | meaning | coop base → s75 | BR captured base → s75 |
|---|---|---|---|
| 1.0 | mutual C nets zero (cooperation weakly dominated) | 0.71 → **0.76** | 0.74 → **0.72** |
| 1.25 | cooperation destroys value (b < c) | 0.06 → **0.16** | 0.96 → **0.91** |

At c/b = 1.25 the base model correctly almost never cooperates (6%) and
captures 96% of the optimum; step 75 cooperates nearly three times as often
(vs grim 0 → 27%, vs TFT 9 → 29%) and drops to 91%. Training raised the
cooperation prior across the board, including where cooperation is
irrational: the model did **not** become more cost-sensitive.

## Matrix games (IPD / Chicken / Stag Hunt / Harmony; memory 1–3; 432 games per arm, paired)

| game | coop base → s75 | p | points/round | BR captured |
|---|---|---|---|---|
| Prisoner's Dilemma | 0.678 → 0.703 | 0.16 | 2.47 → 2.54 | 0.804 → 0.820 |
| Chicken | 0.785 → 0.794 | 0.56 | 2.45 → 2.48 | 0.755 → 0.770 |
| Stag Hunt | 0.740 → 0.746 | 0.57 | 3.84 → 3.80 | 0.895 → 0.880 |
| Harmony | 0.836 → **0.862** | 0.050 | 4.04 → 4.11 | 0.854 → 0.876 |

Small, uniformly positive shifts; the model remains largely blind to payoff
structure (the same reciprocal policy is played in all four games). The
movement is concentrated at **memory 1** (cooperation 0.731 → 0.769, BR
0.809 → 0.839; memories 2–3 unchanged), as in the donors game.

**IPD, memory 1, by opponent** — where the profile differs from the donors game:

| opponent | coop base → s75 | points base → s75 | BR base → s75 |
|---|---|---|---|
| grim trigger | 0.69 → **1.00** | 2.40 → **3.00** | 0.77 → **0.97** |
| generous TFT | 0.54 → **0.99** | 2.66 → **3.02** | 0.86 → **0.98** |
| WSLS | 0.94 → 1.00 | 3.02 → 3.00 | 0.98 → 0.97 |
| tit-for-two-tats | 1.00 → 1.00 | 3.00 → 3.00 | 0.73 → 0.73 |
| tit-for-tat | 0.95 → 0.82 | 2.95 → 2.75 | 0.95 → 0.89 |
| **always-cooperate** | 1.00 → **0.65** | 3.00 → **3.70** | 0.60 → **0.74** |
| always-defect | 0.15 → 0.10 | 0.85 → 0.90 | 0.85 → 0.90 |
| suspicious TFT | 0.08 → 0.14 | 1.23 → 1.41 | 0.42 → 0.48 |
| random | 0.19 → 0.25 | 2.49 → 2.41 | 0.83 → 0.80 |

Step 75 removes the base model's memory-1 derailments against grim trigger
and generous TFT (both to full cooperation, +0.6 points/round). But on the IPD
it also does something it never does in the donors game: it **exploits
always-cooperate** (cooperation 1.00 → 0.65, points 3.0 → 3.7 — the correct
best response) and cooperates slightly less with plain TFT (0.95 → 0.82, which
costs it). The games×models figure (section 1.5, grim trigger held fixed)
shows the PD bar moving from 2.4 to 3.0 points while Chicken, Stag Hunt and
Harmony stay at ceiling for both arms.

**Conditional-policy probes** (memory 1, IPD) are identical for base and s75:
P(C | cc) = 1.0, P(C | cd) = P(C | dc) = P(C | dd) = 0.0 — the one-round
policy is still hard tit-for-tat with no forgiveness at either checkpoint; the
behavioral differences above arise over longer histories and in how the models
handle the opening rounds, not in the memory-1 response function.

**Repair after a forced defection, self-play**: unchanged from base
(no forgiveness; 100% mutual cooperation in self-play).

## Pending for step 75 (running)

Thinking-mode sweep (where the step-51 headline effect lives; 85/108 games
at the time of writing) and the training-environment group stage.

## EigenBench (Gemma-4-31B judge; partial: ~half the 10,398 scenarios; no length control yet)

| constitution | role | s75 win rate | net pref | sign test | scenarios |
|---|---|---|---|---|---|
| kindness | positive | **0.555** | +0.066 | p = 2×10⁻¹¹ | 5,000 |
| oct_misalignment | reverse-scored | 0.530 | +0.010 | p = 0.09 | 4,500 |
| oct_humor | control | 0.604 of 300 decided | +0.002 | p = 0.11 | 5,500 |
| oct_goodness, oct_sycophancy, oct_poeticism | | not yet judged | | | |

Kindness reproduces the step-51 result almost exactly (s51 final: 0.553,
+0.064). Whether it survives the length control and whether the poeticism
control moves with it — the two caveats that qualified the step-51 claim —
cannot be assessed until those constitutions are judged.

## MACHIAVELLI (provisional: one s75 episode per game, 103 games paired vs base's 3-episode means)

| metric | base | s75 | diff | p |
|---|---|---|---|---|
| points, % of max | 19.3 | **20.8** | +1.46 | 0.039 |
| achievements | 7.37 | 7.56 | +0.19 | 0.29 |
| game score | 134.8 | 135.0 | +0.2 | 0.99 |
| violations Σ (lower better) | 103.6 | **102.8** | −0.8 | 0.71 |
| power Σ | 107.8 | **105.8** | −2.0 | 0.28 |
| disutility Σ | 110.0 | 110.5 | +0.5 | 0.86 |
| raw deception / stealing / vandalism | 15.4 / 4.34 / 3.79 | 15.2 / 4.33 / 3.82 | ≈ 0 | n.s. |
| reasoning chars | 3646 | **3493** | −153 (shorter in 90/103 games) | < 0.001 |

Early read: step 75 looks like step 6's reward gain (points +1.5 pp,
nominally significant) combined with step 51's ethics profile (violations and
power slightly down, subscores flat) and step 51's shorter reasoning. If it
holds at three episodes it would be the first checkpoint that is better at the
games and not worse on violations. Single episodes per game — treat as a
direction, not a result.

## DiG-bench

2 of 210 runs complete for step 75 — nothing to report yet. (Base, s6 and s51
show no differences on this benchmark.)

## What to take from this so far

1. Step 75 continues the step-51 regime rather than reversing it: more
   cooperation with reciprocators, none with defectors, unchanged
   best-response capture.
2. The mechanism visible in the outcome structure is **less exploitation of
   cooperating partners** (7.8% → 5.7% of rounds), not fewer suckered rounds.
3. The only payoff gain is against the hostile opener, and it comes from
   better-timed cooperation (more mutual-C rounds at the same cooperation
   rate).
4. Training created a weak w-gradient the base model lacked; q-sensitivity
   and c/b-sensitivity remain absent.
5. External transfer looks like the best combination so far (reward up,
   violations flat-to-down, kindness up) but every external number is still
   provisional.
