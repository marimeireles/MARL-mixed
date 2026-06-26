# Heterogeneous-observation N-player IPD on JaxMARL

This brings the observability experiments from the CRLD study into the
**sampled, deep-RL** regime, so they can be trained with JaxMARL's PPO baselines,
scaled to many agents, and run on GPU.

## Why a separate environment

The CRLD side of this repo runs on **Python 3.8 / jax 0.4.13** (see the repo
root). JaxMARL needs a newer interpreter and JAX, so it lives in its own
virtualenv, created with [`uv`](https://docs.astral.sh/uv/):

```bash
# from the repo root
uv venv --python 3.11 .venv-jaxmarl
uv pip install --python .venv-jaxmarl/bin/python jaxmarl "fastcore>=1.5.27" pydoe
uv pip install --python .venv-jaxmarl/bin/python "jax==0.4.38" "jaxlib==0.4.38"
```

(`jax`/`jaxlib` are pinned to 0.4.38 — the version JaxMARL ships with — because
some transitive deps will otherwise pull a much newer JAX that JaxMARL has not
been tested against. For GPU, install the matching CUDA `jaxlib` wheel.)

Run everything below with that interpreter, e.g.
`../.venv-jaxmarl/bin/python smoke_test.py`.

## Files

| file | what it is |
|---|---|
| `heterogeneous_ipd.py` | the `MultiAgentEnv`: N-player memory-1 IPD with per-agent observation masks |
| `smoke_test.py` | API + game-logic checks (no training); run this first |
| `train_ippo.py` | self-contained feed-forward IPPO (shared policy), returns final cooperation rate |
| `sweep_table.py` | trains every (regime, N) cell and prints the cooperation table |

## The game

N agents, each round choosing **C** or **D**. Reward is the *pairwise-averaged*
Prisoner's Dilemma: agent *i* plays the 2×2 PD (R=1, T=1.2, S=−0.5, P=0) against
every other agent and gets the mean payoff. At **N=2 this is exactly the original
2-agent PD**; the per-capita scale is independent of N, so increasing N changes
only the group size and the observability structure, not the dilemma strength.

Memory-1: each agent conditions on the previous round's joint action, seen
through its observation mask.

## Observability regimes

A regime is an (N, N) visibility matrix `V`, where `V[i,j]=1` iff agent *i* sees
agent *j*'s last action:

| regime | V | meaning |
|---|---|---|
| `full` | all ones | everyone sees everyone |
| `blind` | all zeros | no one sees anything (effectively memoryless) |
| `self` | identity | each agent sees only its own last action |
| `others` | 1 − identity | each agent sees others but not itself |
| `coop` | self + others who cooperated | cooperation-tracking |
| `def` | self + others who defected | defection-tracking |

All six reduce to the corresponding 2-agent conditions from the CRLD study.
`V` is per-row, so **heterogeneous populations** (some agents `self`, others
`full`, …) are a one-line change if you want mixed groups later.

## Usage

```bash
cd jaxmarl_env
PY=../.venv-jaxmarl/bin/python

# 1. sanity-check the env
$PY smoke_test.py

# 2. train one cell
$PY train_ippo.py 4 self          # N=4, self-observation regime

# 3. build the regime x N table (CPU ok for small budgets; GPU for real numbers)
$PY sweep_table.py --agents 2,3,4,5 --regimes full,blind,self,others \
                   --timesteps 5000000 --seeds 3
```

`sweep_table.py` prints a markdown table and writes `coop_table.csv`.

## Notes / next steps

- **Result so far:** independent PPO learners on `full` at N=2 converge to mutual
  defection (cooperation 0.48 → 0.0), matching the CRLD finding that defection is
  the default attractor. To *find* cooperation you need the levers identified on
  the CRLD side (reduce greed, longer horizon, exploration shaping) — JaxMARL's
  finite-sample noise is itself one such lever.
- **Using the official baseline instead of `train_ippo.py`:** register the env
  with `jaxmarl.registration` (map an id to `HeterogeneousIPD`) and point
  `baselines/IPPO/ippo_ff_mpe.py` at it; obs/action dims are read from the env's
  spaces. `train_ippo.py` is provided so the sweep is runnable without editing
  the JaxMARL tree.
- `coop`/`def` regimes are action-conditioned and generalize to any N. The other
  CRLD-specific maskings can be added as new branches in `_visibility`.

## Sweeping every JaxMARL environment (SLURM)

`run_all.py` trains IPPO on one registered JaxMARL env and logs the learning
dynamics (mean reward/step vs. update) to `results/<env>.{npz,png,status}`. It
auto-skips envs it cannot handle with a generic feed-forward, discrete-action
policy and records the reason:

- **continuous actions** (MaBrax: `ant_4x2`, `halfcheetah_6x1`, ...) -> need a
  Gaussian policy head.
- **image observations** above `MAX_OBS_DIM` -> need a CNN encoder.

Everything else (MPE discrete variants, `switch_riddle`, `coin_game`, `storm`,
`hanabi`, SMAX, `overcooked` flattened, ...) trains with the shared FF policy.

```bash
# one env, locally
../.venv-jaxmarl/bin/python run_all.py coin_game --timesteps 5000000

# all 30 envs as a SLURM array (one task per env, 8 concurrent, GPU each)
sbatch slurm_sweep.sh
squeue -u $USER

# after it finishes: coverage report + combined grid figure
../.venv-jaxmarl/bin/python plot_all.py     # -> results/all_dynamics.png
```

`envs.txt` is the env list the array indexes into (regenerate with
`python -c "import jaxmarl,io; open('envs.txt','w').write('\n'.join(jaxmarl.registered_envs))"`).
The job forces the venv's bundled CUDA `ptxas` onto PATH because the compute
nodes' system `ptxas` (12.2) is too old to assemble the PTX that jax 0.4.38
emits.
