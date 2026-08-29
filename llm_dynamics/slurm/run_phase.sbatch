#!/bin/bash
#SBATCH --job-name=llmdyn-phase
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/phase-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/phase-%j.err
#SBATCH --export=ALL
# One phase of the v3 suite, so phases can run in parallel on separate servers.
# Env: TAG, MODEL, ENDPOINT_FILE, PHASE in {horizons, think, perturb_selfplay, group, matrix_probes}
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
TAG="${TAG:?}"; MODEL="${MODEL:?}"; ENDPOINT_FILE="${ENDPOINT_FILE:?}"; PHASE="${PHASE:?}"
STRATS="tit_for_tat suspicious_tit_for_tat always_cooperate always_defect tit_for_two_tats grim_trigger random wsls generous_tit_for_tat"
for _ in $(seq 1 180); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
[ -f "${ENDPOINT_FILE}.ready" ] || { echo "endpoint never became ready"; exit 1; }
API=$(cat "$ENDPOINT_FILE"); echo "phase $PHASE endpoint $API model $MODEL ($(date -Iseconds))"
export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"
COMMON="--api-base $API --model $MODEL"
P=llm_dynamics/results/probes; L=llm_dynamics/logs/phase-${SLURM_JOB_ID}; R=llm_dynamics/results
case "$PHASE" in
  main)   # main direct-play donors sweep: 9 opponents x 3 w x 3 q x 4 memories x 4 seeds
    for s in $STRATS; do
      $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0,0.5,1.0 --w-values 0,0.5,1.0 \
          --strategies $s --memories full,1,2,note2 --seeds 4 --rounds 20 --skip-portraits \
          --out ${TAG}_donors_v3/$s > $L.donors.$s.log 2>&1 &
    done; wait ;;
  cb)     # c/b side sweeps
    for cb in 0.25 0.75 1.0 1.25; do for s in $STRATS; do
      $RUN donors-sweep $COMMON --b 4 --c-over-b $cb --q-values 0,1.0 --w-values 0.5,1.0 \
          --strategies $s --memories full --seeds 4 --rounds 20 --skip-portraits \
          --out ${TAG}_donors_v3_cb${cb}/$s > $L.cb$cb.$s.log 2>&1 &
    done; wait; done ;;
  horizons)
    for n in 5 10; do for s in $STRATS; do
      $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0.5 --w-values 1.0 --strategies $s \
          --memories full --seeds 4 --rounds $n --skip-portraits --out ${TAG}_donors_v3_N$n/$s > $L.N$n.$s.log 2>&1 &
    done; wait; done ;;
  think)
    for s in $STRATS; do
      $RUN donors-sweep $COMMON --thinking --with-think --max-tokens 4096 --b 4 --c-over-b 0.5 \
          --q-values 0,0.5,1.0 --w-values 1.0 --strategies $s --memories full --seeds 4 --rounds 20 \
          --skip-portraits --out ${TAG}_donors_v3_think/$s > $L.think.$s.log 2>&1 &
    done; wait ;;
  think_q)   # thinking sweep for a single q value (env Q), 9 strategy processes
    for s in $STRATS; do
      $RUN donors-sweep $COMMON --thinking --with-think --max-tokens 4096 --b 4 --c-over-b 0.5 \
          --q-values ${Q:?} --w-values 1.0 --strategies $s --memories full --seeds 4 --rounds 20 \
          --skip-portraits --out ${TAG}_donors_v3_think/$s > $L.think.q$Q.$s.log 2>&1 &
    done; wait ;;
  ipd_rep)   # independent replication: IPD, memory 1-3, 20 fresh seeds, separate out dir
    for s in $STRATS; do
      $RUN matrix-sweep $COMMON --games ipd --memories 1,2,3 --strategies $s \
          --seeds 20 --rounds 20 --skip-portraits --out ${TAG}_matrix_v3_rep20/$s > $L.ipdrep.$s.log 2>&1 &
    done; wait ;;
  think_rep)   # independent replication of the reasoning-mode headline: 4 opponents, single q (env Q), 12 fresh seeds
    for s in tit_for_tat generous_tit_for_tat grim_trigger always_cooperate; do
      $RUN donors-sweep $COMMON --thinking --with-think --max-tokens 4096 --b 4 --c-over-b 0.5 \
          --q-values ${Q:?} --w-values 1.0 --strategies $s --memories full --seeds 12 --rounds 20 \
          --skip-portraits --out ${TAG}_donors_v3_think_rep12/$s > $L.thinkrep.q$Q.$s.log 2>&1 &
    done; wait ;;
  fine_w)   # fine sweep in w at q=0.5, c/b=0.5: threshold test for direct reciprocity (w > c/b)
    for s in tit_for_tat grim_trigger generous_tit_for_tat always_cooperate always_defect; do
      $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --q-values 0.5 --w-values 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
          --strategies $s --memories full --seeds 6 --rounds 20 --skip-portraits --out ${TAG}_donors_v3_wfine/$s > $L.wfine.$s.log 2>&1 &
    done; wait ;;
  fine_q)   # fine sweep in q at w=1, c/b=0.5: threshold test for indirect reciprocity (q > c/b)
    for s in tit_for_tat grim_trigger generous_tit_for_tat always_cooperate always_defect; do
      $RUN donors-sweep $COMMON --b 4 --c-over-b 0.5 --w-values 1.0 --q-values 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
          --strategies $s --memories full --seeds 6 --rounds 20 --skip-portraits --out ${TAG}_donors_v3_qfine/$s > $L.qfine.$s.log 2>&1 &
    done; wait ;;
  perturb_selfplay)
    $RUN donors-perturb $COMMON --seeds 4 --rounds 20 --perturb-round 8 --out ${TAG}_donors_v3_perturb > $L.perturb.log 2>&1 &
    $RUN donors-selfplay $COMMON --q-values 0,1.0 --w-values 0,1.0 --seeds 4 --rounds 20 --out ${TAG}_selfplay_v3 > $L.selfplay.log 2>&1 &
    wait ;;
  group)
    $RUN group-eval $COMMON --thinking --with-think --max-tokens 4096 --temperature 0.8 --scenarios 50 --seeds 2 --out ${TAG}_group_v3 > $L.group.log 2>&1 ;;
  matrix_probes)
    for s in $STRATS; do
      $RUN matrix-sweep $COMMON --games ipd,chicken,staghunt,harmony --memories 1,2,3 --strategies $s \
          --seeds 4 --rounds 20 --skip-portraits --out ${TAG}_matrix_v3/$s > $L.matrix.$s.log 2>&1 &
    done; wait
    for m in 1 2 3; do for wq in "0.75 0.75" "0.25 0.25"; do set -- $wq
      $RUN donors-probe $COMMON --w $1 --q $2 --memory $m --out $P/${TAG}_donors_m${m}_w$1_q$2.json > /dev/null &
    done; done; wait
    for g in ipd chicken staghunt harmony; do for m in 1 2 3; do
      $RUN matrix-probe $COMMON --game $g --memory $m --out $P/${TAG}_${g}_m$m.json > /dev/null &
    done; done; wait ;;
  *) echo "unknown PHASE $PHASE"; exit 1 ;;
esac
echo "phase $PHASE done ($(date -Iseconds))"
