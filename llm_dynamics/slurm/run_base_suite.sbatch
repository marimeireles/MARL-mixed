#!/bin/bash
#SBATCH --job-name=llmdyn-qwen32b-base
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-qwen3-32b-base-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-qwen3-32b-base-%j.err
#SBATCH --export=ALL
#
# CPU-only client job: waits for the vLLM endpoint written by
# serve_qwen32b.sbatch, then runs the base-model comparison suite.
#   sbatch llm_dynamics/slurm/run_base_suite.sbatch
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
ENDPOINT_FILE=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm_endpoint.txt
TAG="${TAG:-qwen32b_base}"
MODEL="${MODEL:-qwen3-32b-base}"

for _ in $(seq 1 180); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
[ -f "${ENDPOINT_FILE}.ready" ] || { echo "endpoint never became ready"; exit 1; }
API=$(cat "$ENDPOINT_FILE")
echo "using endpoint $API model $MODEL  ($(date -Iseconds))"
curl -sf -m 10 "$API/v1/models" | head -c 300; echo

export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"
COMMON="--api-base $API --model $MODEL"
P=llm_dynamics/results/probes

echo "== smoke probe =="
$RUN donors-probe $COMMON --w 0.75 --q 0.75 --memory 1 --out $P/${TAG}_donors_m1_w0.75_q0.75.json

echo "== donors probes (w,q corners, memory 1 + memory 2) =="
for wq in "0.25 0.25" "0.25 0.75" "0.75 0.25"; do set -- $wq
  $RUN donors-probe $COMMON --w $1 --q $2 --memory 1 --out $P/${TAG}_donors_m1_w$1_q$2.json
done
$RUN donors-probe $COMMON --w 0.75 --q 0.75 --memory 2 --out $P/${TAG}_donors_m2_w0.75_q0.75.json

echo "== donors q x w sweep =="
$RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 \
    --q-values 0,0.5,1.0 --w-values 0,0.5,1.0 \
    --strategies tit_for_tat,always_defect,always_cooperate \
    --seeds 2 --rounds 20 --out ${TAG}_donors_sweep

echo "== reciprocity figure (base only; add trained probe later) =="
$RUN reciprocity-figure --probes $P/${TAG}_donors_m1_w0.75_q0.75.json \
    --played llm_dynamics/results/${TAG}_donors_sweep/rounds/*tit_for_tat_q0.5_w1_s0.jsonl \
    --opponent tit_for_tat --out llm_dynamics/results/${TAG}_reciprocity_tft.png

echo "== matrix games: memory sweep =="
$RUN matrix-sweep $COMMON --games ipd,chicken,staghunt,harmony --memories 1,2 \
    --strategies tit_for_tat,always_defect --seeds 1 --rounds 20 --out ${TAG}_matrix_sweep
for g in ipd chicken staghunt; do for m in 1 2; do
  $RUN matrix-probe $COMMON --game $g --memory $m --out $P/${TAG}_${g}_m$m.json
done; done
echo "== DONE $(date -Iseconds) =="
