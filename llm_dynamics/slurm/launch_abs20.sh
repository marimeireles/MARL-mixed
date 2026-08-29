#!/bin/bash
# Fully parallel pipeline for the abstract-step20 checkpoint (continuation of step 75
# with 20 extra conversational/naturalistic steps). Everything runs on SLURM.
set -uo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
MODEL=3l3ktr4/donorsim-qwen3-8b-abstract-step20; NAME=qwen3-8b-rl-abs20; TAG=qwen8b_rl_abs20
L=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs
EXTRA="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9"
for _ in $(seq 1 360); do grep -q "^DONE" $L/abs20_download.log 2>/dev/null && break; sleep 20; done
grep -q "^DONE" $L/abs20_download.log || { echo "download did not finish"; exit 1; }
# --- three servers so the dynamics phases, the benchmarks and the think sweep never queue on each other
EPA=$L/vllm_endpoint_${NAME}.txt; EPB=$L/vllm_endpoint_${NAME}-b.txt; EPC=$L/vllm_endpoint_${NAME}-c.txt
SA=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=8030 ENDPOINT_FILE=$EPA MAX_LEN=40960 EXTRA_ARGS="$EXTRA" sbatch --parsable --time=24:00:00 llm_dynamics/slurm/serve_model.sbatch); echo "serverA=$SA"
SB=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=8032 ENDPOINT_FILE=$EPB MAX_LEN=40960 EXTRA_ARGS="$EXTRA" sbatch --parsable --time=24:00:00 llm_dynamics/slurm/serve_model.sbatch); echo "serverB=$SB"
SC=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=8034 ENDPOINT_FILE=$EPC MAX_LEN=40960 EXTRA_ARGS="$EXTRA" sbatch --parsable --time=24:00:00 llm_dynamics/slurm/serve_model.sbatch); echo "serverC=$SC"
# --- dynamics phases, all concurrent, spread over the three servers
sub(){ PHASE=$1 Q=${4:-} TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$2 sbatch --parsable --dependency=after:$3 llm_dynamics/slurm/run_phase.sbatch; }
echo "main=$(sub main $EPA $SA)"; echo "cb=$(sub cb $EPB $SB)"; echo "matrix_probes=$(sub matrix_probes $EPC $SC)"
echo "horizons=$(sub horizons $EPA $SA)"; echo "perturb_selfplay=$(sub perturb_selfplay $EPB $SB)"; echo "group=$(sub group $EPC $SC)"
for pair in "0 $EPA $SA" "0.5 $EPB $SB" "1.0 $EPC $SC"; do set -- $pair
  echo "think q=$1: $(PHASE=think_q Q=$1 TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$2 sbatch --parsable --dependency=after:$3 llm_dynamics/slurm/run_phase.sbatch)"
done
for pair in "0.5 $EPA $SA" "1.0 $EPB $SB"; do set -- $pair
  echo "fine_w/q on $2: $(PHASE=fine_w TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$2 sbatch --parsable --dependency=after:$3 llm_dynamics/slurm/run_phase.sbatch) $(PHASE=fine_q TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$2 sbatch --parsable --dependency=after:$3 llm_dynamics/slurm/run_phase.sbatch)"
done
# replications (same protocol as step 75, so the arms are comparable)
echo "ipd_rep=$(sub ipd_rep $EPA $SA)"
for pair in "0 $EPA $SA" "0.5 $EPB $SB" "1.0 $EPC $SC"; do set -- $pair
  echo "think_rep q=$1: $(PHASE=think_rep Q=$1 TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$2 sbatch --parsable --dependency=after:$3 llm_dynamics/slurm/run_phase.sbatch)"
done
# --- external benchmarks
M=$(MACH_ARMS='{"abs20":"qwen3-8b-rl-abs20"}' EPISODES=3 CONCURRENCY=64 ENDPOINT_FILE=$EPB sbatch --parsable --dependency=after:$SB external_benchmarks/slurm/bench_machiavelli.sbatch); echo "machiavelli=$M"
echo "machiavelli resume=$(MACH_ARMS='{"abs20":"qwen3-8b-rl-abs20"}' EPISODES=3 CONCURRENCY=64 ENDPOINT_FILE=$EPB sbatch --parsable --dependency=afterany:$M external_benchmarks/slurm/bench_machiavelli.sbatch)"
echo "digbench=$(DIGBENCH_API_TOKEN=$(cat ~/.digbench_token) DIG_ARMS='{"abs20":"qwen3-8b-rl-abs20"}' REPS=10 CONCURRENCY=21 ENDPOINT_FILE=$EPC sbatch --parsable --dependency=after:$SC --time=24:00:00 external_benchmarks/slurm/bench_digbench.sbatch)"
E=$(SPEC=runs/qwen8b/spec_abs20.py OUT=runs/qwen8b/responses/responses_rl_abs20.jsonl SCENARIOS=10398 sbatch --parsable external_benchmarks/slurm/bench_eigen_responses.sbatch); echo "eigen responses=$E"
i=0; for c in kindness oct_goodness oct_misalignment oct_sycophancy oct_humor oct_poeticism; do
  G=$([ $((i%2)) -eq 0 ] && echo A100-SXM4-80GB:1 || echo A100-PCI-80GB:1); i=$((i+1))
  echo "judge $c: $(CONSTS="$c" TP=1 GPU_UTIL=0.92 MAX_MODEL_LEN=12288 RL=runs/qwen8b/responses/responses_rl_abs20.jsonl SRC=runs/qwen8b/responses/arms_head_to_head_abs20.jsonl OUT_DIR=runs/qwen8b/judgments_gemma4_abs20 sbatch --parsable --gpus=$G --time=1-00:00:00 --dependency=afterok:$E external_benchmarks/slurm/bench_eigen_judge_gemma4.sbatch)"
done
echo "== abs20 pipeline submitted $(date -Iseconds) =="
