"""Build the cooperation table: rows = observability regime, cols = #agents.

Trains independent PPO learners (shared policy) on HeterogeneousIPD for each
(regime, N) cell and records the final mean cooperation rate. Prints a markdown
table and saves a CSV.

CPU is fine for small budgets/N; use a GPU and a larger TOTAL_TIMESTEPS for
publication-grade numbers. Example:
    python sweep_table.py --agents 2,3,4 --regimes full,blind,self,others --timesteps 3000000 --seeds 3
"""
import argparse, csv, os
import jax
import numpy as np
import train_ippo as t


def run(agents, regimes, timesteps, seeds, num_envs):
    table = {}
    for regime in regimes:
        for N in agents:
            cell = []
            for s in range(seeds):
                cr = t.final_coop_rate(
                    num_agents=N, regime=regime, seed=s,
                    TOTAL_TIMESTEPS=timesteps, NUM_ENVS=num_envs)
                cell.append(cr)
            table[(regime, N)] = (float(np.mean(cell)), float(np.std(cell)))
            print(f"  {regime:7s} N={N}: coop={np.mean(cell):.3f} "
                  f"(+/-{np.std(cell):.3f}, {seeds} seeds)")
    return table


def render(table, agents, regimes, path):
    header = "| regime | " + " | ".join(f"N={N}" for N in agents) + " |"
    sep = "|" + "---|" * (len(agents) + 1)
    lines = [header, sep]
    for regime in regimes:
        row = [regime] + [f"{table[(regime, N)][0]:.2f}" for N in agents]
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    print("\n" + md)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime"] + [f"N={N}" for N in agents])
        for regime in regimes:
            w.writerow([regime] + [f"{table[(regime, N)][0]:.4f}" for N in agents])
    print(f"\nsaved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="2,3,4")
    p.add_argument("--regimes", default="full,blind,self,others")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--num_envs", type=int, default=64)
    a = p.parse_args()
    agents = [int(x) for x in a.agents.split(",")]
    regimes = a.regimes.split(",")
    table = run(agents, regimes, a.timesteps, a.seeds, a.num_envs)
    render(table, agents, regimes,
           os.path.join(os.path.dirname(__file__), "coop_table.csv"))
