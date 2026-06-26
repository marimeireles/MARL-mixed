import os,sys,time
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+"/.."))
import numpy as np
from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from scratch_repro.repro import obs_matrix_from_obs_set, FULL, OSET_A1
np.random.seed(0)
def cc(cond,lr,gamma,beta,n=300):
    env=MultipleObsSocialDilemma(rewards=1,temptations=1.2,suckers_payoffs=-0.5,punishments=0,observation_value=[1,1])
    h=HistoryEmbedded(env,h=(2,2,2)); h.O[0]=obs_matrix_from_obs_set(FULL)
    h.O[1]=obs_matrix_from_obs_set(FULL if cond=="full" else OSET_A1[cond])
    mae=POstratAC(env=h,learning_rates=lr,discount_factors=gamma,choice_intensities=beta)
    mae.obsdist(mae.random_softmax_strategy())
    c=0;nc=0
    for _ in range(n):
        x=mae.random_softmax_strategy();Xt,conv=mae.trajectory(x,Tmax=4000,tolerance=1e-5)
        if conv:
            nc+=1;r=np.asarray(mae.Ri(Xt[-1]))
            if r[0]>0.9 and r[1]>0.9:c+=1
    return 100*c/max(nc,1)
conds=["full","coop_focus","def_focus","self_aware","non_self_aware"]
for tag,(lr,g,b) in {"baseline lr.1":(0.1,0.9,1),"BEST lr.04":(0.04,0.9,1),"lr.04+b2":(0.04,0.9,2)}.items():
    print(f"=== {tag} ===")
    for cn in conds:
        print(f"  {cn:16s} CC={cc(cn,lr,g,b):5.1f}%")
