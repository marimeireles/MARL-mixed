"""Quick probe: which knobs raise the cooperation rate at full observability?
Each config runs N random inits and reports mutual-cooperation rate (both R>0.9)."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from scratch_repro.repro import obs_matrix_from_obs_set, FULL

np.random.seed(0)


def cc_rate(R=1, T=1.2, S=-0.5, P=0, lr=0.1, gamma=0.9, beta=1, n=200, mem=2):
    env = MultipleObsSocialDilemma(rewards=R, temptations=T, suckers_payoffs=S,
                                   punishments=P, observation_value=[1, 1])
    h = HistoryEmbedded(env, h=(mem, mem, mem))
    full = obs_matrix_from_obs_set(FULL) if mem == 2 else None
    if mem == 2:
        h.O[0] = full; h.O[1] = full
    mae = POstratAC(env=h, learning_rates=lr, discount_factors=gamma,
                    choice_intensities=beta)
    mae.obsdist(mae.random_softmax_strategy())
    cc = 0; nconv = 0
    for _ in range(n):
        x = mae.random_softmax_strategy()
        Xt, c = mae.trajectory(x, Tmax=4000, tolerance=1e-5)
        if c:
            nconv += 1
            r = np.asarray(mae.Ri(Xt[-1]))
            if r[0] > 0.9 and r[1] > 0.9:
                cc += 1
    return 100 * cc / max(nconv, 1)


CONFIGS = [
    ("baseline (b=1,g=.9)", dict()),
    ("beta=2", dict(beta=2)),
    ("beta=4", dict(beta=4)),
    ("beta=8", dict(beta=8)),
    ("gamma=.95", dict(gamma=0.95)),
    ("gamma=.98", dict(gamma=0.98)),
    ("alpha=.04", dict(lr=0.04)),
    ("remove greed T=1.0", dict(T=1.0)),
    ("remove fear S=0", dict(S=0.0)),
    ("beta=4+gamma=.98", dict(beta=4, gamma=0.98)),
    ("beta=4+g.98+a.04", dict(beta=4, gamma=0.98, lr=0.04)),
]

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for name, kw in CONFIGS:
        t0 = time.time()
        cc = cc_rate(n=N, **kw)
        print(f"{name:24s} CC={cc:5.1f}%   ({time.time()-t0:.0f}s)")
