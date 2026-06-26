import numpy as np

from pyCRLD.Agents.APOStrategySarsa import stratSARSA as POSARSA
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
import pyCRLD.Utils.FlowPlot as fp

import matplotlib
matplotlib.use("Agg")


def build():
    env = MultipleObsSocialDilemma(
        rewards=1, temptations=1.5, suckers_payoffs=-0.5, punishments=0,
        observation_value=[1, 1],
    )
    memo = HistoryEmbedded(env, h=(1, 1, 1))
    return memo


def run(AgentCls, name):
    memo = build()
    mae = AgentCls(env=memo, learning_rates=0.1, discount_factors=0.9)
    print(f"\n=== {name} ===")
    print("N, Z, M, Q =", mae.N, mae.Z, mae.M, mae.Q)
    np.random.seed(0)
    x0 = mae.random_softmax_strategy()
    print("x0 shape:", x0.shape)
    traj, fixedp = mae.trajectory(x0, Tmax=8000, tolerance=1e-5)
    Xfinal = traj[-1]
    print("converged (fixed point reached):", fixedp)
    print("final policy shape:", Xfinal.shape)
    # cc state is index 0 (both cooperated). action 0 = cooperate.
    print("cooperation prob in (c,c) obs, per agent:", np.array(Xfinal[:, 0, 0]))
    return mae, Xfinal


mae_sarsa, X_sarsa = run(POSARSA, "PO-SARSA")
mae_ac, X_ac = run(POstratAC, "PO-ActorCritic")

print("\n=== Flow plot (PO-SARSA) ===")
fp.plot_strategy_flow(
    mae_sarsa, ([0], [0], [0]), ([1], [0], [0]),
    use_RPEarrows=False, flowarrow_points=np.linspace(0.01, 0.99, 9),
)
print("flow plot ran OK")
