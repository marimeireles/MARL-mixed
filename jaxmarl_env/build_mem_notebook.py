"""Build mem<M>_flow_grids_notebook.ipynb — the observability flow-grid matrix at
memory M (agents condition on the last M rounds). Usage: python build_mem_notebook.py 2
"""
import os, sys
import nbformat as nbf

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2
HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

# The arena (N-player) state space is (2^N)^M, which the CRLD einsum intermediates
# blow past quickly. It is tractable for N=2,3,4 only at memory-1/2; for memory>=3
# we cap the arena at N=2 (N>=3 is too large — see the size note below).
ARENA_NS = [2, 3, 4] if M <= 2 else [2]
GAMES = [("ipd", "Prisoner's Dilemma (IPD)", [2]),
         ("harmony", "Harmony", [2]),
         ("staghunt", "Stag Hunt", [2]),
         ("snowdrift", "Snowdrift / Chicken", [2]),
         ("coin", "coin_game (PD reduction)", [2]),
         ("arena", "STORM arena (N-player)", ARENA_NS)]
ALGOS = [("ac", "CRLD actor-critic  (~ IPPO/A2C)"),
         ("sarsa", "CRLD SARSA  (~ IQL)")]

md(f"""# CRLD observability flow grids — **memory-{M}**

Same six-panel observability grids as the memory-1 notebook, but here every agent
conditions on the **last {M} rounds** (`HistoryEmbedded h=({M},…)`), so the
underlying state space is $(2^N)^{{{M}}}$. The flow is drawn in the state where every
agent cooperated in *all {M}* remembered rounds.

**A note on agent pairs.** All agents are **homogeneous** (the same observability
regime is applied to every agent) and the game is symmetric, so agent 0 ≡ agent 1 ≡
… statistically — every agent-pair projection looks the same. We therefore plot a
single representative pair (**agent 0 vs agent 1**). (Heterogeneous agents — a
different regime per agent — would make the pairs genuinely differ; that is a
separate experiment.)

Axes / reading: arrows = where the learning rule pushes the joint policy; purple =
trajectories; (0,0) = mutual defection, (1,1) = mutual cooperation.""")

code("""import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
plt.rcParams.update({"figure.dpi": 130})
RESULTS = os.path.join(os.getcwd(), "results")
def show(fname, title, figsize=(13, 8)):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p):
        print("(not generated:", fname, ")"); return
    fig, ax = plt.subplots(figsize=figsize); ax.imshow(mpimg.imread(p)); ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", color="#114"); plt.show()
""")

RES = os.path.join(HERE, "results")
skipped = []
for gkey, gname, gns in GAMES:
    cells = []
    for akey, alabel in ALGOS:
        for N in gns:
            fname = f"memgrid_m{M}_{gkey}_{akey}_N{N}.png"
            if os.path.exists(os.path.join(RES, fname)):
                title = f"{gname}  -  {alabel}  -  N={N}  -  memory={M}"
                cells.append(f'show({fname!r}, {title!r})')
            else:
                skipped.append(f"{gkey} {alabel} N={N}")
    if cells:                       # only add the game section if something exists
        md(f"## {gname}")
        for c in cells:
            code(c)
if M >= 3:
    s2, s3, s4 = 4 ** M, 8 ** M, 16 ** M
    md(f"""## Why the arena stops at N=2 here (state-space size)

The N-player arena has $(2^N)^{{{M}}}$ states, and the CRLD einsum *intermediates*
grow faster still, so at memory-{M} only **N=2** is tractable:

| arena | states at memory-{M} | status |
|---|---|---|
| N=2 (shown) | {s2:,} | cheap |
| N=3 (omitted) | {s3:,} | too large — cancelled |
| N=4 (omitted) | {s4:,} | too large |

At memory-4 the arena N=3 case ({8**4:,} states) **OOM-killed at 16 GB and did not
finish even at 96 GB after ~30 min**; at memory-3 ({8**3:,} states) it ran for 20+
min without finishing. Both were cancelled. The 2-player games and arena N=2 are
cheap by comparison.""")
elif skipped:
    md("> **Not shown (state space too large at this memory):** " + "; ".join(skipped) + ".")

md(f"""## What memory-{M} changes (vs memory-1)

Longer memory gives the agents **richer reciprocal strategies** (memory-{M} variants
of tit-for-tat / win-stay-lose-shift). In the *deterministic* CRLD dynamics this can
**enlarge the cooperative basin** — e.g. the full-observability IPD develops a
trajectory that reaches mutual cooperation (1,1) under memory-{M}, which it did not
under memory-1 actor-critic. (Note this is the opposite of the *sampled* deep-RL
result, where extra memory mostly *hurt* because the larger observation space is
harder to learn from finite samples — the deterministic-vs-finite-batch distinction
again.)

**Caveat — why no memory-10 here.** The state space is $(2^N)^{{{M}}}$, so memory grows
the analytic tensors exponentially: 2-player memory-2 = 16 states, memory-4 = 256,
**memory-10 ≈ 1,000,000 states ⇒ tens of TB of tensors — physically infeasible.**
The memory-10 *cooperation* result (a sampled-RL cooperation curve, which avoids the
tensor blow-up) lives in the deep-RL `results_notebook` memory sweep instead.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(HERE, f"mem{M}_flow_grids_notebook.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
