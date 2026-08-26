# Normalization coefficient provenance

Merged file covers **133** games (30 from the MACHIAVELLI release, 103 generated here).

⚠️ 3 working game(s) still have no coefficients: eagles-heir, rebels, werewolves-2-pack-mentality

## Agreement on the 29 overlapping games

Ratio of our random-agent mean to the paper's. 1.00 = identical; our rollouts cap at 400 steps to match `run_arms.py`, theirs do not, so ratios below 1 on cumulative counts are the expected direction.

| count | median ratio | min | max | games within ±10% |
|---|---|---|---|---|
| `game.score` | 0.932 | 0.528 | 1.021 | 17/29 |
| `violations.Σ` | 0.997 | 0.618 | 1.053 | 28/29 |
| `power.Σ` | 0.992 | 0.533 | 1.031 | 28/29 |
| `utility.Σ` | 0.993 | 0.558 | 1.181 | 26/29 |

If these ratios sit near 1.00, mixing provenances is safe and the 30 shipped games stay on the paper's own scale. If they do not, rerun the analysis against `normalization_coeffs.ours.json` for all 136 games instead -- that file is self-consistent by construction.

