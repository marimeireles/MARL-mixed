#!/bin/bash
#SBATCH --job-name=llmdyn-v3-finish
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v3fin-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v3fin-%j.err
#SBATCH --export=ALL
# Finish an interrupted v3 run: rerun matrix sweeps for STRATS, then probes and tables.
set -uo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
TAG="${TAG:?}"; MODEL="${MODEL:?}"; ENDPOINT_FILE="${ENDPOINT_FILE:?}"; STRATS="${STRATS:-}"
API=$(cat "$ENDPOINT_FILE"); echo "endpoint $API model $MODEL ($(date -Iseconds))"
export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"
COMMON="--api-base $API --model $MODEL"; P=llm_dynamics/results/probes; L=llm_dynamics/logs/suite-v3fin-${SLURM_JOB_ID}; R=llm_dynamics/results
echo "== matrix rerun: $STRATS =="
for s in $STRATS; do
  $RUN matrix-sweep $COMMON --games ipd,chicken,staghunt,harmony --memories 1,2,3 --strategies $s \
      --seeds 4 --rounds 20 --skip-portraits --out ${TAG}_matrix_v3/$s > $L.matrix.$s.log 2>&1 &
done; wait
echo "== probes =="
for m in 1 2 3; do for wq in "0.75 0.75" "0.25 0.25"; do set -- $wq
  $RUN donors-probe $COMMON --w $1 --q $2 --memory $m --out $P/${TAG}_donors_m${m}_w$1_q$2.json > $L.probe.d$m.$1.log 2>&1 &
done; done; wait
for g in ipd chicken staghunt harmony; do for m in 1 2 3; do
  $RUN matrix-probe $COMMON --game $g --memory $m --out $P/${TAG}_${g}_m$m.json > $L.probe.$g.$m.log 2>&1 &
done; done; wait
echo "== tables =="
PY=/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python
{ $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3/* $R/${TAG}_matrix_v3/*;
  for cb in 0.25 0.75 1.0 1.25; do echo; echo "## c/b = $cb"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_cb${cb}/*; done;
  echo; echo "## thinking ON (w=1)"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_think/*;
  for n in 5 10; do echo; echo "## horizon N=$n"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_N$n/*; done;
  echo; $PY -m llm_dynamics.analysis signal $R/${TAG}_donors_v3/* $R/${TAG}_donors_v3_perturb; } > $R/${TAG}_v3_tables.md
echo "== DONE $(date -Iseconds) =="
