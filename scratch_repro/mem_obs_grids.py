"""Observability flow grids at HIGHER MEMORY (memory-m).

Same 6-panel observability grids as obs_grids.py / nplayer_obs.py, but the agents
condition on the last `m` rounds (HistoryEmbedded h=(m,)*(N+1)). The observability
masks are applied per memory-round. The flow is drawn in the "all agents cooperated
in every remembered round" state.

State space is (2^N)^m, so this is feasible for memory 2 (and 4 for small N) but
NOT memory 10 (a million+ states -> tens of TB of tensors).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Utils import FlowPlot as fp
from scratch_repro.nplayer_obs import NPlayerSocialDilemma
from scratch_repro.obs_grids import GAMES

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jaxmarl_env", "results"))
REGIMES = [("Full observability", "full"),
           ("Self-aware\n(sees own action)", "self"),
           ("Non-self-aware\n(sees others' actions)", "others"),
           ("Cooperation-tracking\n(sees other iff it cooperated)", "coop"),
           ("Defection-tracking\n(sees other iff it defected)", "def"),
           ("Blind", "blind")]


def _mask_round(round_str, regime, ag, N):
    """Mask the N action slots of one remembered round for agent `ag`'s view."""
    p = round_str.split(",")            # [a0, a1, ..., a_{N-1}, state]
    acts = p[:N]
    out = list(p)
    if regime == "full":
        pass
    elif regime == "blind":
        for j in range(N): out[j] = "."
    elif regime == "self":
        for j in range(N):
            if j != ag: out[j] = "."
    elif regime == "others":
        out[ag] = "."
    elif regime == "coop":              # self always; other j iff j cooperated this round
        for j in range(N):
            if j != ag and acts[j] != "c": out[j] = "."
    elif regime == "def":
        for j in range(N):
            if j != ag and acts[j] != "d": out[j] = "."
    else:
        raise ValueError(regime)
    return ",".join(out)


def _oset_for_mem(regime, base_entry, ag, N):
    rounds = base_entry.strip("|").split("|")
    return "|".join(_mask_round(r, regime, ag, N) for r in rounds) + "|"


def _obs_matrix(descriptions):
    """Row-stochastic obs matrix; rounds separated by '|', slots by ','."""
    n = len(descriptions)
    M = np.zeros((n, n))
    parts = [d.replace("|", ",").strip(",").split(",") for d in descriptions]
    fully = [all(p != "." for p in pp) for pp in parts]
    for i in range(n):
        if fully[i]:
            M[i, i] = 1.0
        else:
            for j in range(n):
                if all(a == "." or a == b for a, b in zip(parts[i], parts[j])):
                    M[i, j] = 1.0
    return M / M.sum(1, keepdims=True)


def _build(game_key, N, memory):
    _, pay = GAMES[game_key]
    if N == 2:
        env = MultipleObsSocialDilemma(rewards=pay["R"], temptations=pay["T"],
                                       suckers_payoffs=pay["S"], punishments=pay["P"],
                                       observation_value=[1, 1])
    else:
        env = NPlayerSocialDilemma(N, pay["R"], pay["T"], pay["S"], pay["P"])
    memo = HistoryEmbedded(env, h=(memory,) * (N + 1))
    return memo


def _allc_state(memo):
    """Index of the state where every remembered round is all-cooperate."""
    for i, lab in enumerate(memo.Sset):
        if "d" not in lab:
            return i
    return 0


def obs_grid(game_key, N, memory, agent_cls=POstratAC, algo_tag="ac",
             algo_label="CRLD actor-critic", NrRandom=16):
    (gname, gsub), pay = GAMES[game_key]
    memo0 = _build(game_key, N, memory)
    si = _allc_state(memo0)
    x = ([0], [si], [0]); y = ([1], [si], [0])
    fig, axes = plt.subplots(2, 3, figsize=(4.4 * 3, 4.3 * 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    for k, (title, reg) in enumerate(REGIMES):
        ax0 = axes[k // 3][k % 3]
        memo = _build(game_key, N, memory)
        base = list(memo.Oset[0])
        for ag in range(N):
            memo.Oset[ag] = [_oset_for_mem(reg, e, ag, N) for e in base]
            memo.O[ag] = _obs_matrix(memo.Oset[ag])
        mae = agent_cls(env=memo, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=NrRandom,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=6000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(title, fontsize=10)
        ax0.set_xlabel("Agent 0  P(cooperate)"); ax0.set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle(f"{gname} {gsub} | {algo_label} | N={N} | MEMORY={memory} | "
                 "observability flow grid", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"memgrid_m{memory}_{game_key}_{algo_tag}_N{N}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[memgrid] m={memory} {game_key} {algo_label} N={N} -> {os.path.basename(out)}")
    return out


_ALGO = {"ac": (POstratAC, "CRLD actor-critic"), "sarsa": (None, "CRLD SARSA")}


def _run_one(game, N, algo, memory):
    from pyCRLD.Agents.APOStrategySarsa import stratSARSA as POSARSA
    cls = POstratAC if algo == "ac" else POSARSA
    obs_grid(game, N, memory, cls, algo, _ALGO[algo][1])


if __name__ == "__main__":
    # Single grid:  python mem_obs_grids.py one <game> <N> <algo> <memory>
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        _, _, game, N, algo, memory = sys.argv
        _run_one(game, int(N), algo, int(memory))
        print("done")
    else:
        # Full set:  python mem_obs_grids.py <memory>
        memory = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        TWO_P = ["ipd", "harmony", "staghunt", "snowdrift", "coin"]
        for algo in ("ac", "sarsa"):
            for g in TWO_P:
                _run_one(g, 2, algo, memory)
            for Nn in [2, 3, 4]:
                _run_one("arena", Nn, algo, memory)
        print("done")
