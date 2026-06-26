"""Run all conditions, save final-reward arrays, print diagnostic breakdown."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scratch_repro.repro import run_condition

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

CONDS = ["full", "self_aware", "non_self_aware", "coop_focus", "def_focus"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def classify(R, hi=0.9, lo=0.1):
    """Joint outcome per converged run from (agent0,agent1) average rewards."""
    a0, a1 = R[:, 0], R[:, 1]
    cc = np.mean((a0 > hi) & (a1 > hi))
    dd = np.mean((a0 < lo) & (a1 < lo))
    exp0 = np.mean((a0 > a1 + 0.3))   # agent0 exploits agent1
    exp1 = np.mean((a1 > a0 + 0.3))   # agent1 exploits agent0
    return cc, dd, exp0, exp1


if __name__ == "__main__":
    for c in CONDS:
        t0 = time.time()
        res = run_condition(c, N)
        R = res["finalR"]
        np.save(os.path.join(OUT, f"finalR_{c}.npy"), R)
        np.save(os.path.join(OUT, f"lengths_{c}.npy"), res["lengths"])
        dt = time.time() - t0
        if len(R):
            cc, dd, e0, e1 = classify(R)
            print(f"[{c:15s}] conv={res['n_converged']}/{N} "
                  f"R0={R[:,0].mean():.3f} R1={R[:,1].mean():.3f} | "
                  f"CC={cc*100:4.1f}% DD={dd*100:4.1f}% "
                  f"exploit0={e0*100:4.1f}% exploit1={e1*100:4.1f}% ({dt:.0f}s)")
        else:
            print(f"[{c}] no converged runs")
    print("saved to", OUT)
