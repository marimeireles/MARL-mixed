#!/bin/bash
# Generalized SLURM array job: render CRLD observability flow grids at a given
# MEMORY, one array task per line of a worklist. CPU-only (pyCRLD analytic).
# Submit e.g.:
#   sbatch --array=1-14%8 --job-name=crld-mem3 \
#          --export=ALL,MEM=3,TASKFILE=jaxmarl_env/mem3_tasks.txt slurm_mem.sh
#
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%a.out
#SBATCH --error=slurm_logs/%x_%a.err

set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed

PY=.venv-jaxmarl/bin/python
: "${MEM:?set MEM (memory length, e.g. 3)}"
: "${TASKFILE:?set TASKFILE (path to worklist)}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASKFILE}")
read -r GAME N ALGO <<< "${LINE}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${GAME} N=${N} ${ALGO} memory=${MEM} on $(hostname)"

export JAX_PLATFORMS=cpu
export PYTHONPATH=/nas/ucb/marimeireles/MARL-mixed
export OMP_NUM_THREADS=4

srun "${PY}" scratch_repro/mem_obs_grids.py one "${GAME}" "${N}" "${ALGO}" "${MEM}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
