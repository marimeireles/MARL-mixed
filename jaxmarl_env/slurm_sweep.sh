#!/bin/bash
# SLURM array job: train IPPO on every registered JaxMARL environment, one array
# task per env, and log the learning dynamics. Submit with:  sbatch slurm_sweep.sh
#
# Each task reads its env name from line $SLURM_ARRAY_TASK_ID of envs.txt and runs
# run_all.py, which trains compatible envs (discrete actions, flat obs) and writes
# a per-env curve + .npz, or a .status file explaining why it was skipped/failed.
#
#SBATCH --job-name=jaxmarl-sweep
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-30%8                  # 30 envs, at most 8 running at once
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err

set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed/jaxmarl_env

PY=../.venv-jaxmarl/bin/python
ENVFILE="${ENVFILE:-envs.txt}"
ENV_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ENVFILE}")
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${ENV_NAME} on $(hostname)"

# --- GPU --------------------------------------------------------------------
# The venv already has CUDA jaxlib installed ("jax[cuda12]==0.4.38"), so each
# task uses the allocated GPU. (If you recreate the venv, reinstall that wheel.)
# Force XLA to use the venv's bundled ptxas (12.9); the system ptxas on the
# compute nodes is 12.2 and cannot assemble the PTX that XLA emits.
export PATH="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

$PY run_all.py "${ENV_NAME}" --timesteps 5000000 --num_envs 64 --seed 0

echo "[$(date)] done ${ENV_NAME}"
