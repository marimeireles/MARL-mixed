#!/bin/bash
#SBATCH --job-name=llmdyn-v2
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v2-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v2-%j.err
#SBATCH --export=ALL
#
# v2 suite: strategies (incl. suspicious TFT) x (w,q) x LLM memory for the
# donors game; strategies x games x memory for matrix games; probes to
# memory 3. Six parallel client streams (one per strategy) against the
# vLLM endpoint from serve_qwen32b.sbatch. CPU-only.
#   TAG=qwen32b_base MODEL=qwen3-32b-base sbatch llm_dynamics/slurm/run_suite_v2.sbatch
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
ENDPOINT_FILE=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm_endpoint.txt
TAG="${TAG:-qwen32b_base}"; MODEL="${MODEL:-qwen3-32b-base}"
STRATS="${STRATS:-tit_for_tat suspicious_tit_for_tat always_cooperate always_defect tit_for_two_tats grim_trigger}"

for _ in $(seq 1 180); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
[ -f "${ENDPOINT_FILE}.ready" ] || { echo "endpoint never became ready"; exit 1; }
API=$(cat "$ENDPOINT_FILE"); echo "endpoint $API model $MODEL ($(date -Iseconds))"
export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"
COMMON="--api-base $API --model $MODEL"
P=llm_dynamics/results/probes
L=llm_dynamics/logs/suite-v2-${SLURM_JOB_ID}

echo "== donors sweep v2 (parallel per strategy) =="
for s in $STRATS; do
  $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0,0.5,1.0 --w-values 0,0.5,1.0 \
      --strategies $s --memories full,1,2,4 --seeds 2 --rounds 20 --skip-portraits \
      --out ${TAG}_donors_v2/$s > $L.donors.$s.log 2>&1 &
done
wait
echo "== matrix sweep v2 (parallel per strategy) =="
for s in $STRATS; do
  $RUN matrix-sweep $COMMON --games ipd,chicken,staghunt,harmony --memories 1,2,3,4,6 \
      --strategies $s --seeds 2 --rounds 20 --out ${TAG}_matrix_v2/$s > $L.matrix.$s.log 2>&1 &
done
wait
echo "== probes to memory 3 =="
for m in 1 2 3; do
  $RUN donors-probe $COMMON --w 0.75 --q 0.75 --memory $m --out $P/${TAG}_donors_m${m}_w0.75_q0.75.json > /dev/null &
done; wait
for g in ipd chicken staghunt harmony; do for m in 1 2 3; do
  $RUN matrix-probe $COMMON --game $g --memory $m --out $P/${TAG}_${g}_m$m.json > /dev/null &
done; done; wait
echo "== tables =="
.venv-jaxmarl/bin/python -m llm_dynamics.analysis table llm_dynamics/results/${TAG}_donors_v2/* llm_dynamics/results/${TAG}_matrix_v2/* \
    | tee llm_dynamics/results/${TAG}_v2_tables.md
echo "== DONE $(date -Iseconds) =="
