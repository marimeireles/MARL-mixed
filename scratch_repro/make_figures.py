"""Generate all paper figures from saved sweep results."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))
os.makedirs(FIG, exist_ok=True)

BLUE, GOLD = "#0254a3", "#fcbc04"
plt.rcParams.update({"font.size": 11})


def load(cond):
    return np.load(os.path.join(RES, f"finalR_{cond}.npy"))


def reward_hist(cond, title, fname):
    R = load(cond)
    pc0 = 100 * np.mean(R[:, 0] > 0.9)
    pc1 = 100 * np.mean(R[:, 1] > 0.9)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist([R[:, 0], R[:, 1]], bins=24, range=(-0.6, 1.3), log=True,
            color=[BLUE, GOLD],
            label=[f"Agent 1 (full): {pc0:.1f}%", f"Agent 2 (partial): {pc1:.1f}%"])
    ax.set_title(title)
    ax.set_xlabel("Average reward per agent")
    ax.set_ylabel("Count (log scale)")
    ax.legend(title="% runs with reward > 0.9", fontsize=8, title_fontsize=8,
              loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)
    return pc0, pc1, R[:, 0].mean(), R[:, 1].mean()


def summary_bar(conds, labels, fname):
    cc, r0, r1 = [], [], []
    for c in conds:
        R = load(c)
        cc.append(100 * np.mean((R[:, 0] > 0.9) & (R[:, 1] > 0.9)))
        r0.append(R[:, 0].mean()); r1.append(R[:, 1].mean())
    x = np.arange(len(conds))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    a1.bar(x, cc, color="#2a8c5a")
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=25, ha="right")
    a1.set_ylabel("Mutual-cooperation rate (%)")
    a1.set_title("(a) Cooperation vs. observability")
    w = 0.38
    a2.bar(x - w/2, r0, w, color=BLUE, label="Agent 1 (full observer)")
    a2.bar(x + w/2, r1, w, color=GOLD, label="Agent 2 (partial observer)")
    a2.set_xticks(x); a2.set_xticklabels(labels, rotation=25, ha="right")
    a2.set_ylabel("Mean average reward")
    a2.set_title("(b) Reward by observational capability")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)
    return cc, r0, r1


def reward_trajectory(fname):
    """Representative reward-over-time trajectory under full observability."""
    from scratch_repro.repro import make_agents
    np.random.seed(7)
    mae = make_agents("full")
    mae.obsdist(mae.random_softmax_strategy())
    x = mae.random_softmax_strategy()
    Xtisa, _ = mae.trajectory(x, Tmax=5000, tolerance=1e-7)
    Rti = np.array([np.asarray(mae.Ri(X)) for X in Xtisa[::5]])
    t = np.arange(len(Rti)) * 5
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t, Rti[:, 0], color=BLUE, label="Agent 1 (full observer)")
    ax.plot(t, Rti[:, 1], color=GOLD, label="Agent 2 (partial observer)")
    ax.set_xlabel("Learning step"); ax.set_ylabel("Average reward")
    ax.set_title("Reward dynamics under mixed observability")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(reward_hist("non_self_aware", "Non-Self-Aware Agent 2", "non_self_aware.png"))
    print(reward_hist("self_aware", "Self-Aware Agent 2", "self_aware.png"))
    print(reward_hist("coop_focus", "Agent 2 Tracks Cooperation", "cooperation_focus.png"))
    print(reward_hist("def_focus", "Agent 2 Tracks Defection", "defection_focus.png"))
    print(reward_hist("full", "Full Observability", "full_obs.png"))
    print(summary_bar(
        ["full", "coop_focus", "def_focus", "self_aware", "non_self_aware"],
        ["Full", "Coop-focus", "Def-focus", "Self-aware", "Non-self-aware"],
        "summary.png"))
    reward_trajectory("reward_trajectory.png")
    print("figures ->", FIG)
