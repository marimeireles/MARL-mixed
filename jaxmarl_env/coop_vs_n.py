"""How does observational heterogeneity x group size affect cooperation?

Trains IPPO on the N-player heterogeneous-observation IPD for every observability
regime and several group sizes N, recording the final mean cooperation rate
(averaged over agents and the last few updates). Saves a regime x N table and
heatmap.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heterogeneous_ipd import REGIMES
import coop_trajectories as ct

RESULTS = os.path.join(os.path.dirname(__file__), "results")
NS = [2, 3, 4, 6]
CFG = dict(ct.CFG, TOTAL_TIMESTEPS=1_500_000, NUM_ENVS=64)


def final_coop(regime, N, seed=0):
    t = ct.coop_trajectory(regime, num_agents=N, seed=seed, cfg=CFG)  # (updates, N)
    tail = max(1, len(t) // 10)
    return float(t[-tail:].mean())          # mean over agents + last 10% of updates


if __name__ == "__main__":
    regimes = list(REGIMES)
    M = np.zeros((len(regimes), len(NS)))
    for i, r in enumerate(regimes):
        for j, N in enumerate(NS):
            M[i, j] = final_coop(r, N)
            print(f"  {r:7s} N={N}: coop={M[i, j]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "coop_vs_n.npy"), M)
    np.save(os.path.join(RESULTS, "coop_vs_n_axes.npy"),
            np.array([regimes, NS], dtype=object), allow_pickle=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_yticks(range(len(regimes))); ax.set_yticklabels(regimes)
    for i in range(len(regimes)):
        for j in range(len(NS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title("Final cooperation rate (IPPO)\nobservability regime x group size")
    fig.colorbar(im, label="mean P(cooperate)")
    fig.tight_layout()
    out = os.path.join(RESULTS, "coop_vs_n.png")
    fig.savefig(out, dpi=150); print("saved", out)
