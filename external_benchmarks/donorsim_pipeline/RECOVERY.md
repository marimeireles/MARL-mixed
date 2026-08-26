# Rebuilding the benchmark environment from scratch

`/capstor/scratch/cscs/mmeireles` is **not backed up and gets purged**. It was purged
once around 2026-08-06 (which destroyed the container image, the base weights, and the
MACHIAVELLI checkout + data) and again on 2026-08-16. Everything in this repository is
recoverable; everything listed under "Not preserved" below has to be re-fetched with the
commands here.

Nothing in this file requires anything that was on scratch.

## What is preserved in this repo

| | where |
|---|---|
| Aggregated metrics (all figures derive from these) | `external-benchmarks/data/*.csv` |
| Executed notebooks | `external-benchmarks/notebooks/*.ipynb` |
| Raw run artifacts, compressed | `external-benchmarks/raw/*.tar.zst` |
| Every run, serving and analysis script | `external-benchmarks/scripts/` |
| Container + model build logs | `provenance/` |

## Not preserved — re-fetch with these commands

### 1. Base weights (~60 GB)

```bash
export HF_HOME=$SCRATCH/hf_cache
hf download Qwen/Qwen3-32B --local-dir $SCRATCH/models/Qwen3-32B-base
```

### 2. The GRPO LoRA adapters

Published on the Hub, so they survive independently of scratch. All 7 checkpoints
(`step_25` … `step_175`) are there; `step_175` is the evaluated arm.

```bash
hf download 3l3ktr4/qwen3-32b-donorsim-loras \
    --local-dir $SCRATCH/models/qwen3-32b-donorsim-loras
```

**Then re-apply this patch**, or vLLM will refuse to load the adapter. The shipped
`adapter_config.json` still points at the training node's ephemeral cache path:

```bash
python - <<'EOF'
import json, pathlib
p = pathlib.Path("<...>/qwen3-32b-donorsim-loras/step_175/adapter_config.json")
p.with_suffix(".json.orig").write_text(p.read_text())        # keep a backup
cfg = json.loads(p.read_text())
# was: /dev/shm/verl-cache/d5cd8eb9b9457cff09dddafaa7f7fdd3/Qwen_Qwen3-32B
cfg["base_model_name_or_path"] = "<...>/models/Qwen3-32B-base"
cfg["inference_mode"] = True
p.write_text(json.dumps(cfg, indent=2))
EOF
```

### 3. The vLLM container image (~16 GB squashfs)

`enroot import docker://` **fails on Lustre** — it cannot create opaque overlayfs
whiteouts (`Operation not permitted`; see `provenance/enroot-import.log`). Go through
podman instead:

```bash
podman pull docker.io/vllm/vllm-openai:latest
enroot import -x mount -o $SCRATCH/ce-images/vllm-openai.sqsh \
    podman://docker.io/vllm/vllm-openai:latest
```

`enroot import` may exit non-zero on temp-directory cleanup while still having written a
valid image. Verify rather than trusting the exit code:

```bash
unsquashfs -s $SCRATCH/ce-images/vllm-openai.sqsh | head   # expect a valid superblock
```

Then restore `~/.edf/` from `external-benchmarks/scripts/serving/vllm.toml`. **Do not
drop the `TRITON_CACHE_DIR` line** — `/dev/shm` is mounted `noexec` in this image, so
Triton cannot map the `cuda_utils*.so` it generates and engine init dies during
compilation.

### 4. MACHIAVELLI

```bash
git clone https://github.com/aypan17/machiavelli.git
cd machiavelli
# game_data.zip is linked from the repo README; the password is: machiavelli
unzip -P machiavelli game_data.zip
```

`game_data/normalization_coeffs.json` is inside that zip and is **required** by
`Trajectory.get_mach_scores()` — a partial extraction that misses it fails at scoring
time, not at load time.

Then restore our agent:

```bash
cp external-benchmarks/scripts/machiavelli_vllm_agent.py machiavelli/agent/vllm_agent.py
cp external-benchmarks/scripts/machiavelli_run_arms.py  run_arms.py
git apply external-benchmarks/scripts/machiavelli_load_agent.patch   # registers VLLMAgent
```

### 5. DiG-bench

```bash
git clone https://github.com/discos-research/dig-bench.git
cd dig-bench && uv venv && uv pip install -e baseline-harness
```

Games are served API-side; the harness ships none of them. You need a DiG-bench API
token in `$DIGBENCH_API_TOKEN`. **The token is deliberately not in this repository** —
mint one at https://digbench.ai and keep it in a `chmod 600` file outside the repo.

Then restore `external-benchmarks/scripts/digbench_orchestrate.py` and the sweep
launchers.

### 6. Python environment for the analysis

```bash
uv venv && uv pip install pandas numpy scipy matplotlib nbformat nbclient ipykernel
```

## Slurm facts worth not rediscovering

- Account is `aa004` (not `a-a04`).
- The account holds **only QoS `normal`**, so the `low` partition (24 h) is rejected
  with `Invalid qos specification`. Max allocation is therefore **12 h** on `normal`,
  and both sweeps take longer than that — which is why both launchers self-chain across
  allocations and both orchestrators are resumable.
- One GH200 node = 4 GPUs × 97871 MiB. Both arms fit in one vLLM process at TP=4.

## Measured throughput (for planning)

| | rate | notes |
|---|---|---|
| DiG-bench | 19.1 runs / node-hour | concurrency 21, ~1.03 h per run |
| MACHIAVELLI | ~41 trajectories / node-hour | concurrency 16, ~23 min per trajectory |

## Restoring the raw artifacts

```bash
zstd -dc external-benchmarks/raw/digbench-runs.tar.zst | tar xf -
zstd -dc external-benchmarks/raw/machiavelli-trajectories.tar.zst | tar xf -
```

Both orchestrators treat restored artifacts as completed work — a DiG-bench run whose
JSONL carries a `summary` row and a MACHIAVELLI episode with an `ep{N}.done.json` marker
are both skipped. So unpacking these into a fresh scratch resumes the sweeps exactly
where they stopped, rather than redoing ~130 node-hours.
