#!/bin/bash
#SBATCH --job-name=llmdyn-v3
#SBATCH --partition=main
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v3-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-v3-%j.err
#SBATCH --export=ALL
# v3 suite (CPU client). Env: TAG, MODEL (served name), ENDPOINT_FILE.
#   TAG=qwen8b_base MODEL=qwen3-8b-base ENDPOINT_FILE=.../vllm_endpoint_qwen3-8b-base.txt sbatch llm_dynamics/slurm/run_suite_v3.sbatch
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
TAG="${TAG:?}"; MODEL="${MODEL:?}"; ENDPOINT_FILE="${ENDPOINT_FILE:?}"
STRATS="tit_for_tat suspicious_tit_for_tat always_cooperate always_defect tit_for_two_tats grim_trigger random wsls generous_tit_for_tat"
for _ in $(seq 1 180); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
[ -f "${ENDPOINT_FILE}.ready" ] || { echo "endpoint never became ready"; exit 1; }
API=$(cat "$ENDPOINT_FILE"); echo "endpoint $API model $MODEL ($(date -Iseconds))"
export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"
COMMON="--api-base $API --model $MODEL"
P=llm_dynamics/results/probes; L=llm_dynamics/logs/suite-v3-${SLURM_JOB_ID}
R=llm_dynamics/results

echo "== donors c/b=0.5, memories full/1/2/note2, 4 seeds =="
for s in $STRATS; do
  $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0,0.5,1.0 --w-values 0,0.5,1.0 \
      --strategies $s --memories full,1,2,note2 --seeds 4 --rounds 20 --skip-portraits \
      --out ${TAG}_donors_v3/$s > $L.donors.$s.log 2>&1 &
done; wait
echo "== donors c/b sweep {0.25,0.75,1.0,1.25} at w in {0.5,1}, q in {0,1} =="
for cb in 0.25 0.75 1.0 1.25; do for s in $STRATS; do
  $RUN donors-sweep $COMMON --b 4 --c-over-b $cb --q-values 0,1.0 --w-values 0.5,1.0 \
      --strategies $s --memories full --seeds 4 --rounds 20 --skip-portraits \
      --out ${TAG}_donors_v3_cb${cb}/$s > $L.cb$cb.$s.log 2>&1 &
done; wait; done
echo "== horizon N in {5,10} at w=1 =="
for n in 5 10; do for s in $STRATS; do
  $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0.5 --w-values 1.0 --strategies $s \
      --memories full --seeds 4 --rounds $n --skip-portraits --out ${TAG}_donors_v3_N$n/$s > $L.N$n.$s.log 2>&1 &
done; wait; done
echo "== thinking ON subset (w=1) =="
for s in $STRATS; do
  $RUN donors-sweep $COMMON --thinking --with-think --max-tokens 4096 --b 4 --c-over-b 0.5 \
      --q-values 0,0.5,1.0 --w-values 1.0 --strategies $s --memories full --seeds 4 --rounds 20 \
      --skip-portraits --out ${TAG}_donors_v3_think/$s > $L.think.$s.log 2>&1 &
done; wait
echo "== perturbation / repair =="
$RUN donors-perturb $COMMON --seeds 4 --rounds 20 --perturb-round 8 --out ${TAG}_donors_v3_perturb > $L.perturb.log 2>&1
echo "== self-play (base vs base) =="
$RUN donors-selfplay $COMMON --q-values 0,1.0 --w-values 0,1.0 --seeds 4 --rounds 20 --out ${TAG}_selfplay_v3 > $L.selfplay.log 2>&1
echo "== group-stage eval (thinking on) =="
$RUN group-eval $COMMON --thinking --with-think --max-tokens 4096 --temperature 0.8 --scenarios 50 --seeds 2 --out ${TAG}_group_v3 > $L.group.log 2>&1
echo "== matrix games, memory 1-3, 4 seeds =="
for s in $STRATS; do
  $RUN matrix-sweep $COMMON --games ipd,chicken,staghunt,harmony --memories 1,2,3 --strategies $s \
      --seeds 4 --rounds 20 --skip-portraits --out ${TAG}_matrix_v3/$s > $L.matrix.$s.log 2>&1 &
done; wait
echo "== probes =="
for m in 1 2 3; do for wq in "0.75 0.75" "0.25 0.25"; do set -- $wq
  $RUN donors-probe $COMMON --w $1 --q $2 --memory $m --out $P/${TAG}_donors_m${m}_w$1_q$2.json > /dev/null &
done; done; wait
for g in ipd chicken staghunt harmony; do for m in 1 2 3; do
  $RUN matrix-probe $COMMON --game $g --memory $m --out $P/${TAG}_${g}_m$m.json > /dev/null &
done; done; wait
echo "== tables =="
PY=/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python
{ $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3/* $R/${TAG}_matrix_v3/*;
  for cb in 0.25 0.75 1.0 1.25; do echo; echo "## c/b = $cb"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_cb${cb}/*; done;
  echo; echo "## thinking ON (w=1)"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_think/*;
  for n in 5 10; do echo; echo "## horizon N=$n"; $PY -m llm_dynamics.analysis table $R/${TAG}_donors_v3_N$n/*; done;
  echo; $PY -m llm_dynamics.analysis signal $R/${TAG}_donors_v3/* $R/${TAG}_donors_v3_perturb; } > $R/${TAG}_v3_tables.md
echo "== DONE $(date -Iseconds) =="
