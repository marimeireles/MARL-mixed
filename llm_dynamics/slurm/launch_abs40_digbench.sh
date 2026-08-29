#!/bin/bash
# DiG-bench for abstract-step40, sharded 5 ways over 5 dedicated servers.
set -uo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
MODEL=3l3ktr4/donorsim-qwen3-8b-abstract-step40; NAME=qwen3-8b-rl-abs40d
L=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs
EXTRA="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9"
TOK=$(tr -d '\r\n' < ~/.digbench_token)
for i in 0 1 2 3 4; do
  EP=$L/vllm_endpoint_${NAME}-$i.txt
  S=$(MODEL_PATH=$MODEL SERVED_NAME=$NAME PORT=$((8060+2*i)) ENDPOINT_FILE=$EP MAX_LEN=40960 EXTRA_ARGS="$EXTRA" sbatch --parsable --time=24:00:00 llm_dynamics/slurm/serve_model.sbatch)
  J=$(SHARD=$i NSHARD=5 AGGREGATE=0 DIGBENCH_API_TOKEN=$TOK DIG_ARMS="{\"abs40\":\"$NAME\"}" REPS=10 CONCURRENCY=8 ENDPOINT_FILE=$EP sbatch --parsable --dependency=after:$S --time=24:00:00 external_benchmarks/slurm/bench_digbench.sbatch)
  R=$(SHARD=$i NSHARD=5 AGGREGATE=0 DIGBENCH_API_TOKEN=$TOK DIG_ARMS="{\"abs40\":\"$NAME\"}" REPS=10 CONCURRENCY=8 ENDPOINT_FILE=$EP sbatch --parsable --dependency=afterany:$J --time=24:00:00 external_benchmarks/slurm/bench_digbench.sbatch)
  echo "shard$i: server=$S client=$J resume=$R"
done
echo "== abs40 DiG submitted (5 shards, 21 games x 10 reps) $(date -Iseconds) =="
