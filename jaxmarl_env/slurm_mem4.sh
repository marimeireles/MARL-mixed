#!/bin/bash
# SLURM array job: render the memory-4 CRLD observability flow grids, one array
# task per (game, N, algo) read from mem4_tasks.txt. Submit with: sbatch slurm_mem4.sh
#
# These are CPU-only analytic (pyCRLD) jobs — no GPU. Each task writes one
# memgrid_m4_<game>_<algo>_N<N>.png into results/.
#
#SBATCH --job-name=crld-mem4
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=1-14%8                   # 14 grids, at most 8 at once
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err

set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed

PY=.venv-jaxmarl/bin/python
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jaxmarl_env/mem4_tasks.txt)
read -r GAME N ALGO <<< "${LINE}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${GAME} N=${N} ${ALGO} memory=4 on $(hostname)"

export JAX_PLATFORMS=cpu                 # analytic CRLD, CPU only
export PYTHONPATH=/nas/ucb/marimeireles/MARL-mixed
export OMP_NUM_THREADS=4

srun "${PY}" scratch_repro/mem_obs_grids.py one "${GAME}" "${N}" "${ALGO}" 4
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
