#!/bin/bash
# Launch the full step-51 benchmark pipeline once the merged model is in the HF cache.
# Everything runs on SLURM; this script only submits jobs.
set -uo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
MODEL=3l3ktr4/donorsim-qwen3-8b-modeAB-step51; NAME=qwen3-8b-rl-s51; TAG=qwen8b_rl_s51
EP=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm_endpoint_${NAME}.txt
for _ in $(seq 1 360); do grep -q "^DONE" llm_dynamics/logs/step51_download.log 2>/dev/null && break; sleep 30; done
grep -q "^DONE" llm_dynamics/logs/step51_download.log || { echo "download did not finish"; exit 1; }
# 1) server (24h): merged step-51 model, Qwen3 parsers (DiG/MACHIAVELLI need them)
S=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=8014 MAX_LEN=40960 EXTRA_ARGS="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9" sbatch --parsable --time=24:00:00 llm_dynamics/slurm/serve_model.sbatch); echo "server=$S"
# 2) dynamics v3 suite + extras/finish safety net
V=$(TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$EP sbatch --parsable --dependency=after:$S llm_dynamics/slurm/run_suite_v3.sbatch); echo "v3 suite=$V"
X=$(TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$EP sbatch --parsable --dependency=afterany:$V llm_dynamics/slurm/finish_extras.sbatch); echo "extras=$X"
F=$(TAG=$TAG MODEL=$NAME ENDPOINT_FILE=$EP STRATS="" sbatch --parsable --dependency=afterany:$V llm_dynamics/slurm/finish_v3.sbatch); echo "finish=$F"
# 3) MACHIAVELLI (rl51 arm) + resume; DiG (rl51 arm)
M=$(MACH_ARMS='{"rl51":"qwen3-8b-rl-s51"}' EPISODES=3 CONCURRENCY=64 ENDPOINT_FILE=$EP sbatch --parsable --dependency=after:$S external_benchmarks/slurm/bench_machiavelli.sbatch); echo "machiavelli=$M"
M2=$(MACH_ARMS='{"rl51":"qwen3-8b-rl-s51"}' EPISODES=3 CONCURRENCY=64 ENDPOINT_FILE=$EP sbatch --parsable --dependency=afterany:$M external_benchmarks/slurm/bench_machiavelli.sbatch); echo "machiavelli resume=$M2"
D=$(DIGBENCH_API_TOKEN=$(cat ~/.digbench_token) DIG_ARMS='{"rl51":"qwen3-8b-rl-s51"}' REPS=10 CONCURRENCY=21 ENDPOINT_FILE=$EP sbatch --parsable --dependency=after:$S --time=24:00:00 external_benchmarks/slurm/bench_digbench.sbatch); echo "digbench=$D"
# 4) EigenBench responses (in-process vLLM, GPU) then Gemma-4 judging vs base (3 parallel jobs)
E=$(SPEC=runs/qwen8b/spec_rl51.py OUT=runs/qwen8b/responses/responses_rl_step51.jsonl SCENARIOS=10398 sbatch --parsable external_benchmarks/slurm/bench_eigen_responses.sbatch); echo "eigen responses=$E"
for pair in "kindness oct_goodness" "oct_misalignment oct_sycophancy" "oct_humor oct_poeticism"; do
  J=$(CONSTS="$pair" TP=1 GPU_UTIL=0.92 MAX_MODEL_LEN=12288 RL=runs/qwen8b/responses/responses_rl_step51.jsonl SRC=runs/qwen8b/responses/arms_head_to_head_step51.jsonl OUT_DIR=runs/qwen8b/judgments_gemma4_step51 sbatch --parsable --gpus=A100-SXM4-80GB:1 --time=2-00:00:00 --dependency=afterok:$E external_benchmarks/slurm/bench_eigen_judge_gemma4.sbatch); echo "judge [$pair]=$J"
done
echo "== step51 pipeline submitted $(date -Iseconds) =="
