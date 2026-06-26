"""Compare IPPO / A2C / IQL cooperation behaviour across observability regimes.

Saves:
  results/algo_curves_full.npy   per-algo cooperation-over-training (full regime)
  results/algo_final.npy         final cooperation, algos x regimes
  results/algo_compare.png       curves + grouped-bar figure
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heterogeneous_ipd import REGIMES
import algorithms as A

RESULTS = os.path.join(os.path.dirname(__file__), "results")
ALGOS = list(A.ALGOS)          # ["IPPO","A2C","IQL"]
REGS = list(REGIMES)
CFG = dict(A.CFG, TOTAL_TIMESTEPS=1_000_000, NUM_ENVS=64)
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}


if __name__ == "__main__":
    final = np.zeros((len(ALGOS), len(REGS)))
    curves = {}
    for ai, name in enumerate(ALGOS):
        fn = A.ALGOS[name]
        for ri, reg in enumerate(REGS):
            t = fn(reg, cfg=CFG)                 # (updates, nA)
            final[ai, ri] = t[-max(1, len(t)//10):].mean()
            if reg == "full":
                curves[name] = t.mean(1)         # mean over agents
            print(f"  {name:5s} {reg:7s}: {final[ai, ri]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "algo_final.npy"), final)
    np.save(os.path.join(RESULTS, "algo_final_axes.npy"),
            np.array([ALGOS, REGS], dtype=object), allow_pickle=True)
    np.save(os.path.join(RESULTS, "algo_curves_full.npy"),
            np.array([curves[n] for n in ALGOS]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ALGOS:
        y = curves[name]
        a1.plot(np.arange(len(y)) * CFG["NUM_STEPS"] * CFG["NUM_ENVS"], y,
                color=COLORS[name], label=name, lw=2)
    a1.set_xlabel("environment steps"); a1.set_ylabel("cooperation P(C)")
    a1.set_title("(a) Cooperation over training — full observability")
    a1.set_ylim(-0.03, 1.03); a1.legend()

    w = 0.25; xs = np.arange(len(REGS))
    for ai, name in enumerate(ALGOS):
        a2.bar(xs + (ai - 1) * w, final[ai], w, color=COLORS[name], label=name)
    a2.set_xticks(xs); a2.set_xticklabels(REGS, rotation=20, ha="right")
    a2.set_ylabel("final cooperation P(C)"); a2.set_ylim(0, 1.03)
    a2.set_title("(b) Final cooperation by regime & algorithm"); a2.legend()
    fig.suptitle("Algorithm comparison on the heterogeneous-observation IPD (N=2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS, "algo_compare.png"), dpi=150)
    print("saved", os.path.join(RESULTS, "algo_compare.png"))
