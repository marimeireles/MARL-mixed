"""Learning-rate sweep: final cooperation vs learning rate, per algorithm,
on the full-observability IPD. Connects to the CRLD finding that a smaller
learning rate enlarges the cooperative basin."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import algorithms as A

RESULTS = os.path.join(os.path.dirname(__file__), "results")
LRS = [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
ALGOS = list(A.ALGOS)
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}

if __name__ == "__main__":
    M = np.zeros((len(ALGOS), len(LRS)))
    for ai, name in enumerate(ALGOS):
        for li, lr in enumerate(LRS):
            cfg = dict(A.CFG, TOTAL_TIMESTEPS=1_000_000, NUM_ENVS=64, LR=lr)
            t = A.ALGOS[name]("full", cfg=cfg)
            M[ai, li] = t[-max(1, len(t)//10):].mean()
            print(f"  {name:5s} lr={lr:.0e}: {M[ai, li]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "lr_sweep.npy"), M)
    np.save(os.path.join(RESULTS, "lr_sweep_axes.npy"),
            np.array([ALGOS, LRS], dtype=object), allow_pickle=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ai, name in enumerate(ALGOS):
        ax.plot(LRS, M[ai], "o-", color=COLORS[name], label=name, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("learning rate")
    ax.set_ylabel("final cooperation P(C)"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Cooperation vs learning rate (full-obs IPD, N=2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "lr_sweep.png"), dpi=150)
    print("saved", os.path.join(RESULTS, "lr_sweep.png"))
