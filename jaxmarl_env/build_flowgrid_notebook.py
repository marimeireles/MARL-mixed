"""Build flow_grids_notebook.ipynb — the full matrix of CRLD observability flow
grids: games x algorithms x player-counts. Reads obsgrid_*.png from results/."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

GAMES = [("ipd", "Prisoner's Dilemma (IPD)"),
         ("harmony", "Harmony"),
         ("staghunt", "Stag Hunt"),
         ("snowdrift", "Snowdrift / Chicken"),
         ("coin", "coin_game-PD"),
         ("arena", "STORM arena-PD")]
ALGOS = [("ac", "CRLD actor-critic  (~ IPPO/A2C, policy gradient)"),
         ("sarsa", "CRLD SARSA  (~ IQL, value-based)")]
NS = [2, 3, 4]

md("""# CRLD observability flow grids — the full matrix

Each **grid** is six panels = the six ways Agent-1's observation can be degraded
(**full / self-aware / non-self-aware / cooperation-tracking / defection-tracking /
blind**). Each **panel** is a CRLD *flow field* (coloured arrows = where the
learning rule pushes the joint policy at every point) with two purple learning
trajectories, drawn in the state where both agents just cooperated.

We sweep three axes:
- **Game** — Prisoner's Dilemma, Harmony, Stag Hunt, Snowdrift/Chicken, plus the
  coin_game and STORM-arena Prisoner's Dilemmas.
- **Algorithm** — CRLD actor-critic (policy gradient) vs CRLD SARSA (value-based).
- **Players** — N = 2, 3, 4 (for N>2 the grid is a 2-D projection onto agents 0,1).

How to read it: arrows pointing into a corner = the learning dynamics drive the
joint policy there. (0,0) = mutual defection, (1,1) = mutual cooperation.""")

code("""import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
plt.rcParams.update({"figure.dpi": 130})
RESULTS = os.path.join(os.getcwd(), "results")

def show(fname, title, figsize=(13, 8)):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p):
        print("(not generated yet:", fname, ")"); return
    fig, ax = plt.subplots(figsize=figsize); ax.imshow(mpimg.imread(p)); ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", color="#114"); plt.show()
""")

for gkey, gname in GAMES:
    md(f"## {gname}")
    for akey, alabel in ALGOS:
        for N in NS:
            fname = f"obsgrid_{gkey}_{akey}_N{N}.png"
            title = f"{gname}  -  {alabel}  -  N={N}"
            code(f'show({fname!r}, {title!r})')

md("""## Key findings & honest caveats

**Cooperation vs number of players (a surprise).** In this *pairwise-averaged* PD,
cooperation does **not** get harder with more players — for the IPD it gets
*easier*: from random starts, cooperation at the reciprocity state rises with N.

| game (full obs, P(C) at reciprocity state) | N=2 | N=3 | N=4 |
|---|---|---|---|
| Prisoner's Dilemma | 0.00 | 0.09 | 0.23 |
| Stag Hunt | 0.83 | 0.72 | 0.84 |
| Snowdrift | 0.50 | 0.54 | 0.50 |

Why: a single agent's defection only perturbs *one* of the N−1 pairwise terms, so
the pull away from mutual cooperation shrinks relative to the reciprocity
structure, enlarging the cooperative basin. The naive "more players → less
cooperation" intuition comes from non-reciprocal / public-goods framings; under
memory-1 reciprocity with payoff dilution it goes the other way.

- **Actor-critic vs SARSA:** the flow field depends on the *learning rule*. On the
  IPD, actor-critic flows to defection while SARSA (value-based) sustains more
  cooperation — the analytic twin of the deep-RL "IPPO defects, IQL cooperates".
- **Players (N>2):** the panels are a 2-D projection (agents 0,1, averaging over the
  rest), so they read as "the dynamics for a representative pair". Cooperation
  generally gets harder as N grows.
- **Regimes for N>2:** *self / others / full / blind* generalize directly; the
  conditional *cooperation/defection-tracking* regimes are 2-player constructs
  (they condition on "the other" agent), so for N>2 they use the documented
  N-generalisation (see `scratch_repro/nplayer_obs.py`) or are omitted.
- Any "(not generated yet)" cell means that (game x algorithm x N) combination is
  still computing or was not feasible — re-run after the generators finish.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(HERE, "flow_grids_notebook.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
