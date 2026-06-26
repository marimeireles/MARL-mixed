"""N-player CRLD social-dilemma environments with observability masking.

Builds the same 6-panel observability flow grids as `obs_grids.py` (which is
2-player only) but for arbitrary N. The game is the *pairwise-averaged*
Prisoner's-Dilemma-family matrix game: agent i's reward is the mean over the
other N-1 agents j of the 2x2 payoff g(a_i, a_j).  At N=2 this is exactly the
2-agent SocialDilemma; for larger N it keeps the per-capita payoff scale fixed.

  g(c,c)=R   g(c,d)=S
  g(d,c)=T   g(d,d)=P                (row = focal action, col = opponent action)

Single state (Z=1), 2 actions.  We then memory-1 embed with
HistoryEmbedded(env, h=(1,)*(N+1)) and override the per-agent observation
tensors to realise the observability regimes for N players.

For N>2 the flow plot is a 2-D PROJECTION onto agents 0 and 1
(x=([0],[s],[0]), y=([1],[s],[0])); plot_strategy_flow averages over the other
agents' strategies via NrRandom.

Run on CPU:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv-jaxmarl/bin/python scratch_repro/nplayer_obs.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Environments.HeterogeneousObservationsEnv import HeterogeneousObservationsEnv
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Utils import FlowPlot as fp

from scratch_repro.obs_grids import GAMES   # reuse (R,T,S,P) payoff dict

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jaxmarl_env", "results"))


# --------------------------------------------------------------------------- #
#  N-player single-state social dilemma (pairwise-averaged PD)                 #
# --------------------------------------------------------------------------- #
class NPlayerSocialDilemma(HeterogeneousObservationsEnv):
    """N-agent, 2-action, single-state social dilemma.

    Reward of agent i = mean_{j != i} g(a_i, a_j), with g the 2x2 (R,T,S,P)
    payoff.  Reduces to the 2-agent SocialDilemma at N=2.
    """

    def __init__(self, N, R, T, S, P,
                 observation_type="default", observation_value=None):
        self.N_players = int(N)
        self.R, self.T_pay, self.S, self.P = float(R), float(T), float(S), float(P)

        # needed before super().__init__ builds the tensors
        self.n_agents = self.N_players
        self.n_agent_actions = 2
        self.n_states = 1
        self.state = 0
        if observation_value is None:
            observation_value = [1.0] * self.n_agents
        super().__init__(observation_type=observation_type,
                         observation_value=observation_value)

    def _g(self, ai, aj):
        # action 0 = cooperate, 1 = defect
        return [[self.R, self.S], [self.T_pay, self.P]][ai][aj]

    def transition_tensor(self):
        N = self.n_agents
        dims = [self.n_states] + [self.n_agent_actions] * N + [self.n_states]
        return np.ones(dims)  # single absorbing state -> stay forever

    def reward_tensor(self):
        N = self.n_agents
        dims = [N, self.n_states] + [self.n_agent_actions] * N + [self.n_states]
        Risas = np.zeros(dims)
        for index, _ in np.ndenumerate(Risas):
            i = index[0]
            jA = index[2:-1]                       # joint action tuple
            others = [jA[j] for j in range(N) if j != i]
            Risas[index] = np.mean([self._g(jA[i], aj) for aj in others])
        return Risas

    def actions(self):
        return [["c", "d"] for _ in range(self.n_agents)]

    def states(self):
        return ["."]

    def id(self):
        return f"NPlayerSocialDilemma_N{self.n_agents}_{self.R}_{self.T_pay}_{self.S}_{self.P}"


# --------------------------------------------------------------------------- #
#  Observability masks generalised to N players                               #
# --------------------------------------------------------------------------- #
def _oset_for_n(regime, base, ag, N):
    """Agent `ag`'s masked observation labels for `regime`, N players.

    Each base label is 'a0,a1,...,a_{N-1},state|'.  A masked slot becomes '.'.
    Regimes:
      full   : see every agent's last action
      self   : see only own action
      others : see every *other* agent's action, not own
      blind  : see nothing
      coop   : own + each other agent j only if j cooperated  (N-version of the
               2-player 'cooperation-tracking' regime)
      def    : own + each other agent j only if j defected
    """
    out = []
    for s in base:
        p = s.strip("|").split(",")            # [a0,...,a_{N-1}, state]
        acts = p[:N]
        q = list(p)                            # keep state slot as-is
        if regime == "full":
            pass
        elif regime == "self":
            for k in range(N):
                if k != ag:
                    q[k] = "."
        elif regime == "others":
            q[ag] = "."
        elif regime == "blind":
            for k in range(N):
                q[k] = "."
        elif regime == "coop":
            for k in range(N):
                if k != ag and acts[k] != "c":
                    q[k] = "."
        elif regime == "def":
            for k in range(N):
                if k != ag and acts[k] != "d":
                    q[k] = "."
        else:
            raise ValueError(regime)
        out.append(",".join(q) + "|")
    return out


def _obs_matrix(descriptions):
    """Observation tensor row for a partial observer: uniform over the states
    that look identical to it (all non-masked coordinates agree).  N-general:
    just splits on ',' and compares every slot.  (Same logic as flowplots.py.)"""
    n = len(descriptions)
    M = np.zeros((n, n))
    parts = [d.strip("|").split(",") for d in descriptions]
    fully = [all(p != "." for p in pp) for pp in parts]
    for i in range(n):
        if fully[i]:
            M[i, i] = 1.0
        else:
            for j in range(n):
                if all(a == "." or a == b for a, b in zip(parts[i], parts[j])):
                    M[i, j] = 1.0
    return M / M.sum(1, keepdims=True)


def apply_regime(memo, regime, N):
    """Override every agent's Oset/O in the memory-embedded env to realise the
    homogeneous observability `regime` (all N agents masked the same way)."""
    base = list(memo.Sset)                     # canonical memory-state labels
    for ag in range(N):
        memo.Oset[ag] = _oset_for_n(regime, base, ag, N)
        memo.O[ag] = _obs_matrix(memo.Oset[ag])
    return memo


REGIMES_FULL = [("Full observability", "full"),
                ("Self-aware\n(sees own action)", "self"),
                ("Non-self-aware\n(sees others' actions)", "others"),
                ("Cooperation-tracking\n(sees agent j iff j cooperated)", "coop"),
                ("Defection-tracking\n(sees agent j iff j defected)", "def"),
                ("Blind", "blind")]


def build(game_key, N):
    (_, _), pay = GAMES[game_key]
    env = NPlayerSocialDilemma(N=N, R=pay["R"], T=pay["T"],
                               S=pay["S"], P=pay["P"])
    memo = HistoryEmbedded(env, h=(1,) * (N + 1))
    return env, memo


# --------------------------------------------------------------------------- #
#  Flow grids                                                                  #
# --------------------------------------------------------------------------- #
def obs_grid(game_key, N, regimes=REGIMES_FULL, NrRandom=24,
             agent_cls=POstratAC, algo_tag="ac", algo_label="CRLD actor-critic"):
    (gname, gsub), pay = GAMES[game_key]
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.3 * nrow))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    x = ([0], [0], [0]); y = ([1], [0], [0])     # agents 0,1 P(C) in all-C state
    for k, (title, reg) in enumerate(regimes):
        ax0 = axes[k // ncol][k % ncol]
        _, memo = build(game_key, N)
        apply_regime(memo, reg, N)
        mae = agent_cls(env=memo, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=NrRandom,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=8000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(title, fontsize=10)
        ax0.set_xlabel("Agent 0  P(cooperate)")
        ax0.set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle(f"{game_key} | {algo_label} | N={N} | "
                 f"observability flow grid (agents 0,1 projection)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"obsgrid_{game_key}_{algo_tag}_N{N}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[obsgrid] {game_key} {algo_label} N={N} -> {os.path.basename(out)}")
    return out


# --------------------------------------------------------------------------- #
#  Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def verify_reduction():
    """N-player env reduces to the 2-agent game at N=2."""
    pay = GAMES["ipd"][1]
    env = NPlayerSocialDilemma(N=2, **{k: pay[k] for k in "RTSP"})
    R = env.R   # [i, s=0, a0, a1, s'=0]
    c, d = 0, 1
    checks = {
        "CC->R (agent0)": (R[0, 0, c, c, 0], pay["R"]),
        "DD->P (agent0)": (R[0, 0, d, d, 0], pay["P"]),
        "agent0 defects unilaterally ->T": (R[0, 0, d, c, 0], pay["T"]),
        "agent0 cooperates, other defects ->S": (R[0, 0, c, d, 0], pay["S"]),
        "agent1 defects unilaterally ->T": (R[1, 0, c, d, 0], pay["T"]),
        "agent1 cooperates, other defects ->S": (R[1, 0, d, c, 0], pay["S"]),
    }
    print("[verify] N=2 reduction to 2-agent SocialDilemma:")
    ok = True
    for name, (got, exp) in checks.items():
        good = np.isclose(got, exp)
        ok &= good
        print(f"   {'OK ' if good else 'BAD'}  {name}: got {got:+.3f} exp {exp:+.3f}")
    # N=3 spot check: one defector among (d,c,c) for agent0 = mean(g(d,c),g(d,c)) = T
    env3 = NPlayerSocialDilemma(N=3, **{k: pay[k] for k in "RTSP"})
    R3 = env3.R
    print("[verify] N=3 spot checks (pairwise-averaged):")
    print(f"   all-C agent0 = {R3[0,0,c,c,c,0]:+.3f} (exp R={pay['R']})")
    print(f"   all-D agent0 = {R3[0,0,d,d,d,0]:+.3f} (exp P={pay['P']})")
    print(f"   agent0 D vs (C,C) = {R3[0,0,d,c,c,0]:+.3f} (exp T={pay['T']})")
    print(f"   agent0 C vs (D,D) = {R3[0,0,c,d,d,0]:+.3f} (exp S={pay['S']})")
    print(f"   agent0 C vs (C,D) = {R3[0,0,c,c,d,0]:+.3f} "
          f"(exp mean(R,S)={0.5*(pay['R']+pay['S']):+.3f})")
    return ok


def converged_cooperation(game_key, N, seeds=6, regime="full"):
    """Converged P(cooperate), averaged over seeds (random initial strategies).

    Returns (allC, allC_std, overall, overall_std) where
      allC    = P(C) in the post-mutual-cooperation state (state 0), the
                reciprocity slice the flow grids plot, averaged over agents;
      overall = P(C) averaged over all agents *and* all 2^N memory states.
    """
    allc, over = [], []
    for seed in range(seeds):
        _, memo = build(game_key, N)
        apply_regime(memo, regime, N)
        mae = POstratAC(env=memo, learning_rates=0.1, discount_factors=0.9)
        np.random.seed(seed)
        xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=8000, tolerance=1e-5)
        Xfin = np.asarray(xt[-1])               # (N, Q, M); action 0 = cooperate
        allc.append(float(Xfin[:, 0, 0].mean()))
        over.append(float(Xfin[:, :, 0].mean()))
    return (float(np.mean(allc)), float(np.std(allc)),
            float(np.mean(over)), float(np.std(over)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    verify_reduction()

    print("\n[verify] converged cooperation (full observability):")
    print("   game      N   P(C|all-C state)     P(C overall, all states)")
    for g in ["ipd", "staghunt", "snowdrift"]:
        for N in (2, 3, 4):
            ac, acs, ov, ovs = converged_cooperation(g, N, seeds=6, regime="full")
            print(f"   {g:9s} {N}   {ac:.3f} +/- {acs:.3f}        {ov:.3f} +/- {ovs:.3f}")

    games = ["ipd", "harmony", "staghunt", "snowdrift", "coin", "arena"]
    figs = []
    for N in (3, 4):
        for g in games:
            figs.append(obs_grid(g, N))
    print("\n[done] figures:")
    for f in figs:
        print("  ", f)
