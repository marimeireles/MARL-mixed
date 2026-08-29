#!/bin/bash
# MACHIAVELLI only, for abstract-step40, sharded 4 ways over 4 dedicated servers.
set -uo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
MODEL=3l3ktr4/donorsim-qwen3-8b-abstract-step40; NAME=qwen3-8b-rl-abs40
L=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs
EXTRA="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9"
for _ in $(seq 1 360); do grep -q "^DONE" $L/abs40_download.log 2>/dev/null && break; sleep 20; done
grep -q "^DONE" $L/abs40_download.log || { echo "download did not finish"; exit 1; }
PORT=8040
for i in 0 1 2 3; do
  EP=$L/vllm_endpoint_${NAME}-$i.txt
  S=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=$((PORT+2*i)) ENDPOINT_FILE=$EP MAX_LEN=40960 EXTRA_ARGS="$EXTRA" sbatch --parsable --time=12:00:00 llm_dynamics/slurm/serve_model.sbatch)
  echo "server$i=$S"
  J=$(SHARD=$i NSHARD=4 AGGREGATE=0 MACH_ARMS="{\"abs40\":\"$NAME\"}" EPISODES=3 CONCURRENCY=24 ENDPOINT_FILE=$EP sbatch --parsable --dependency=after:$S external_benchmarks/slurm/bench_machiavelli.sbatch)
  echo "  shard$i=$J"
  R=$(SHARD=$i NSHARD=4 AGGREGATE=0 MACH_ARMS="{\"abs40\":\"$NAME\"}" EPISODES=3 CONCURRENCY=24 ENDPOINT_FILE=$EP sbatch --parsable --dependency=afterany:$J external_benchmarks/slurm/bench_machiavelli.sbatch)
  echo "  shard$i resume=$R"; LAST=$R
done
echo "== abs40 MACHIAVELLI submitted (4 shards x 34 games x 3 episodes) $(date -Iseconds) =="
