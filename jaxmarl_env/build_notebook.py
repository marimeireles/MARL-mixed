"""Assemble results_notebook.ipynb from the saved sweep artifacts (nbformat).
The notebook only reads results/*.npz, *.status and the CRLD figures, so it runs
without importing jax/jaxmarl."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# MARL-mixed — multi-agent learning dynamics

This notebook summarizes the experiments added to **MARL-mixed**:

1. **JaxMARL sweep** — IPPO trained on *every registered JaxMARL environment*
   (one SLURM array task per game, GPU each), logging the learning dynamics.
2. **CRLD observability study** — the 2-agent / N-agent Iterated Prisoner's
   Dilemma with heterogeneous observability (deterministic learning dynamics).

Everything below is generated from saved artifacts in `results/` — no training
or GPU needed to re-run the notebook.""")

code("""import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath("__file__")) if "__file__" in dir() else os.getcwd()
RESULTS = os.path.join(HERE, "results")
REPO = os.path.dirname(HERE)
pd.set_option("display.max_rows", 100)
print("results dir:", RESULTS)""")

md("""## 1. Coverage across all JaxMARL environments

Each environment produced a `.status` file (`ok` / `skipped:<reason>` /
`failed:<reason>`). A single feed-forward, **discrete-action** IPPO trains the
discrete games directly; continuous-action games need a Gaussian policy head
(added separately) and a few image/dict-observation games need a CNN encoder.""")

code("""rows = []
for s in sorted(glob.glob(os.path.join(RESULTS, "*.status"))):
    env = os.path.basename(s)[:-7]
    st = open(s).read().strip()
    kind = st.split(":", 1)[0]
    reason = st.split(":", 1)[1] if ":" in st else ""
    rows.append(dict(environment=env, status=kind, reason=reason))
cov = pd.DataFrame(rows)
summary = cov["status"].value_counts().rename("count").to_frame()
print(summary)
cov""")

md("## 2. Learning dynamics of the trained games\n\n"
   "Reward per step vs. environment steps. Most games show clear improvement; "
   "hard exploration games (SMAX) stay near flat at this budget.")

code("""trained = []
for f in sorted(glob.glob(os.path.join(RESULTS, "*.npz"))):
    d = np.load(f)
    env = os.path.basename(f)[:-4]
    y = d["dynamics"]
    trained.append(dict(environment=env,
                        n_agents=int(d["num_agents"]),
                        reward_start=round(float(y[0]), 3),
                        reward_end=round(float(y[-1]), 3),
                        delta=round(float(y[-1] - y[0]), 3),
                        updates=len(y)))
perf = pd.DataFrame(trained).sort_values("delta", ascending=False).reset_index(drop=True)
perf""")

code("""files = sorted(glob.glob(os.path.join(RESULTS, "*.npz")))
ncol = 4; nrow = (len(files) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 2.6*nrow), squeeze=False)
for ax in axes.flat: ax.axis("off")
for k, f in enumerate(files):
    d = np.load(f); ax = axes[k//ncol][k%ncol]; ax.axis("on")
    ax.plot(d["timesteps"], d["dynamics"], color="#0254a3")
    ax.set_title(f"{os.path.basename(f)[:-4]} (N={int(d['num_agents'])})", fontsize=9)
    ax.set_xlabel("env steps", fontsize=8); ax.set_ylabel("reward/step", fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle("IPPO learning dynamics across JaxMARL environments", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()""")

md("""### Skipped & failed — and why

- **skipped: continuous action space** — MaBrax robotic-control envs and the
  FACMAC MPE variants use `Box` actions; handled by the Gaussian-policy run
  (Section 4).
- **failed** — environment-specific API quirks (dict/image observations, a
  `make()` that needs extra args). Each is a localized follow-up.""")

code("""notok = cov[cov.status != "ok"][["environment", "status", "reason"]]
notok.reset_index(drop=True)""")

md("""## 3. Continuous control (MaBrax) — Gaussian policy

The discrete sweep above skips `Box`-action games. A Gaussian policy head lets
IPPO train the MaBrax robotic-control suite (`ant_4x2`, `halfcheetah_6x1`,
`hopper_3x1`, `humanoid_9|8`, `walker2d_2x3`). Their curves are loaded below once
that run has completed (look for `*_mabrax` / continuous entries in the table
above).

*This section auto-populates from the same `results/` directory, so re-running
the notebook after the MaBrax sweep finishes will show the new curves with no
code changes.*""")

code("""mabrax = [r for r in trained if r["environment"] in
          {"ant_4x2","halfcheetah_6x1","hopper_3x1","humanoid_9|8","walker2d_2x3"}]
if mabrax:
    display(pd.DataFrame(mabrax))
else:
    print("MaBrax continuous results not present yet — run the Gaussian sweep, "
          "then re-execute this notebook.")""")

md("""## 4. CRLD observability study (the original research question)

Before the deep-RL sweep, the core study uses **collective reinforcement
learning dynamics** (CRLD, the deterministic learning-dynamics limit) on the
Iterated Prisoner's Dilemma with *heterogeneous observability*: one agent sees
the full interaction, the other is selectively blind.

Key reproduced finding: **cooperation is fragile** — defection is the modal
attractor, and impairing observability suppresses cooperation monotonically;
losing the ability to observe one's *own* last action is uniquely catastrophic.""")

code("""summ = os.path.join(REPO, "paper", "figures", "summary.png")
if os.path.exists(summ):
    from matplotlib import image as mpimg
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.imshow(mpimg.imread(summ)); ax.axis("off")
    ax.set_title("CRLD: cooperation rate & reward vs. observability (2-agent IPD)")
    plt.show()
else:
    print("summary figure not found:", summ)""")

code("""# N-agent extension: cooperation rate as a function of #agents x regime
csv = os.path.join(HERE, "coop_table.csv")
if os.path.exists(csv):
    display(pd.read_csv(csv))
else:
    print("coop_table.csv not found")""")

md("""## 5. Cooperation dynamics — phase portraits

For the games where **cooperation** is genuinely defined (the mixed-motive
social dilemma), we can draw the *dynamics of cooperation* directly, in the
style of pyCRLD's `fp.plot_trajectories`: train IPPO while logging each agent's
cooperation frequency P(play C) per update, then trace the joint policy's path
through the **(agent-1 cooperation, agent-2 cooperation)** plane.

Below: the heterogeneous-observation memory-1 IPD (N=2) under each observability
regime. `x` marks the (random) start, `o` the converged policy. Under every
regime the trajectory collapses toward mutual defection (0, 0) — the deep-RL
echo of the CRLD result that defection is the dominant attractor; the regimes
differ in *how* they get there and how much residual cooperation survives.

*(Most JaxMARL games are forced-cooperative or non-dilemma, so a C/D cooperation
axis is not defined for them — for those, the reward curves in Section 2 are the
learning dynamics.)*""")

code("""coop = sorted(glob.glob(os.path.join(RESULTS, "coop_*.npy")))
if coop:
    names = {"full":"Full obs","blind":"Blind","self":"Self only","others":"Others only",
             "coop":"Coop-tracking","def":"Def-tracking"}
    order = ["full","others","coop","def","self","blind"]
    have = {os.path.basename(f)[5:-4]: np.load(f) for f in coop}
    regs = [r for r in order if r in have] + [r for r in have if r not in order]
    ncol = 3; nrow = (len(regs)+ncol-1)//ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5*ncol, 3.3*nrow), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for k, r in enumerate(regs):
        t = have[r]; ax = axes[k//ncol][k%ncol]; ax.axis("on")
        ax.plot(t[:,0], t[:,1], lw=2, color="purple", alpha=0.9)
        ax.scatter(t[0,0], t[0,1], color="purple", marker="x", s=45)
        ax.scatter(t[-1,0], t[-1,1], color="purple", marker="o", s=45)
        ax.plot([0,1],[0,1], ls=":", c="gray", lw=0.7)
        ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.03,1.03)
        ax.set_title(names.get(r,r), fontsize=10)
        ax.set_xlabel("Agent 1  P(C)"); ax.set_ylabel("Agent 2  P(C)")
    fig.suptitle("Cooperation dynamics across observability regimes (IPD, N=2)", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97]); plt.show()
    pd.DataFrame([{"regime":r,
                   "start": tuple(np.round(have[r][0],2)),
                   "end": tuple(np.round(have[r][-1],2))} for r in regs])
else:
    print("coop_*.npy not present yet — run coop_trajectories.py")""")

md("""## 6. What was added to this repository

| path | what it is |
|---|---|
| `jaxmarl_env/heterogeneous_ipd.py` | N-player memory-1 IPD as a JaxMARL `MultiAgentEnv`, 6 observability regimes |
| `jaxmarl_env/train_ippo.py` | shared-policy feed-forward IPPO for the IPD (discrete) |
| `jaxmarl_env/run_all.py` | generic IPPO driver: trains one JaxMARL env, logs dynamics (discrete **and** Gaussian-continuous) |
| `jaxmarl_env/slurm_sweep.sh` | SLURM array job — one task per registered env |
| `jaxmarl_env/plot_all.py` | coverage report + combined dynamics grid |
| `jaxmarl_env/sweep_table.py` | regime x N cooperation table for the IPD |
| `jaxmarl_env/results/` | per-env `.npz` curves, `.png`, `.status` |
| `scratch_repro/` | CRLD reproduction scripts (deterministic IPD dynamics) |
| `paper/` | corrected write-up + regenerated figures |
| `.venv-jaxmarl/` | Python 3.11 env (uv) with JaxMARL + CUDA jax 0.4.38 (git-ignored) |

Two interpreters are used deliberately: the CRLD code runs on the repo's
Python 3.8 / jax 0.4.13; the JaxMARL code runs in `.venv-jaxmarl` (Python 3.11).""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_notebook.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
