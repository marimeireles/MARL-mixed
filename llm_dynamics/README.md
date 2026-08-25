# llm_dynamics — cooperation dynamics of LLM agents

Applies this repo's methodology (CRLD flow fields + measured learning
trajectories, see `jaxmarl_env/algo_phase.py` and the paper) to two new
kinds of learner:

1. the **donorSim donors game** in its basic dyadic form — one LLM vs one
   set strategy, sweeping `q`, `w`, `c/b` — with prompts that exactly
   mirror the `direct_indirect` GRPO training rollouts of
   donorSim@neurips-methodology, so RL-trained checkpoints are evaluated
   in-distribution;
2. **classic matrix games** (PD, Chicken, Stag Hunt, Harmony) played by an
   LLM under an explicit memory-m constraint.

The point of the design: the RL training in donorSim (GRPO + normalized
payoff + HKB coordination reward) is claimed to *change the dynamics* of
the LLM. Here that claim becomes a picture — base model and trained
checkpoint as two trajectories / two policy points in the same phase
portrait, over the CRLD flow that predicts what a small adaptive learner
would do.

## The theory mapping (donors game → CRLD)

| donorSim | CRLD |
|---|---|
| payoffs `(b, c)` | donation-game PD `(R,T,S,P) = (b−c, b, −c, 0)`; on the training's normalized Eq.-2 scale `(b/(b+c), 1, 0, c/(b+c))` |
| `w` (re-encounter probability) | discount factor `γ` (Nowak's `w > c/b` ⇔ discounted-repeated-game threshold) |
| `q` (reputation availability) | observability blend `O = q·O_full + (1−q)·O_self` of the paper's `full`/`self` regimes |
| per-partner memory | `HistoryEmbedded(env, h=(m,m,m))` |
| set strategy opponent | pinned policy `X_opp[s,a]`; the LLM side gets a **fixed-opponent flow field** |

All of this lives in `donors_crld.py` (built on `pyCRLD`, reusing the
observability machinery of `scratch_repro/mem_obs_grids.py`).

## The measurement

* **Continuous order parameter.** `p_cooperate` is read from the top-k
  logprobs at the token after the final `DECISION:` marker (renormalized
  COOPERATE-mass vs DEFECT-mass), following donorSim's
  `metastability_eval/harness/vllm_client.py`. Binary actions hide weak
  restoring/repelling tendencies; this doesn't.
* **Cooperation plane** — sliding-window (model P(C), partner P(C)) paths
  over the two-agent CRLD flow (grey = deterministic theory, colour =
  measured LLM; `x` start, `o` end). Deviation from the flow is the finding.
* **Reciprocity plane** — the LLM's conditional policy
  `(P(C | sustained mutual C), P(C | own C, partner D))`, measured two ways:
  directly by **probing** every memory-m history state (`policy_probe.py`,
  a point per model), and behaviorally from played games (windowed
  conditional frequencies). Background: the fixed-opponent CRLD flow.
* **q × w heatmaps** with Nowak's `q = c/b` threshold marked.
* Per-round JSONL logs also carry the HKB phase `φ` (window 4, same
  bookkeeping as the training reward), payoffs on both scales, and swap
  events — compatible with donorSim's metastability analyses.

## Faithfulness to the training environment

`donors_game.py` reproduces the `direct_indirect` rollout of
`donor_sim/verl_integration/donor_game_interaction.py`: identical initial
prompt (rules, w/q phrasing, q-gated reputation report, `/no_think`),
identical round-result turns, simultaneous-move semantics (partner reacts
to prior rounds only), and the w-gated partner swap with history + HKB
window reset. One deliberate deviation: `--swap-mode same` (default)
keeps the successor partner on the *same* strategy so the dyad stays
controlled; `--swap-mode pool` reproduces the training distribution.
Opponent strategies, payoffs, HKB, and parsing are vendored verbatim in
`strategies.py`.

`matrix_games.py` instead makes every round a *stateless* call carrying
only the last m rounds — the LLM is then genuinely a memory-m policy on
exactly the state space of `HistoryEmbedded(SocialDilemma(R,T,S,P),
h=(m,m,m))`, so the CRLD comparison is exact.

## Running

Everything runs from the repo root in the `.venv-jaxmarl` venv:

```bash
RUN="JAX_PLATFORMS=cpu .venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"

# offline smoke test of the whole pipeline (mock base vs mock trained):
eval $RUN demo

# dyadic donors game vs a served model (vLLM/SGLang, OpenAI-compatible):
eval $RUN donors-sweep --api-base http://host:8000 --model Qwen/Qwen3-8B \
    --b 4 --c-over-b 0.5 --q-values 0,0.25,0.5,0.75,1.0 \
    --w-values 0,0.25,0.5,0.75,1.0 \
    --strategies tit_for_tat,always_defect,always_cooperate \
    --seeds 3 --rounds 40 --out donors_qwen_base

# probe the conditional policy of base vs RL-trained checkpoints:
eval $RUN donors-probe --api-base http://host:8000 --model Qwen/Qwen3-8B \
    --w 0.75 --q 0.75 --out llm_dynamics/results/probes/base.json
eval $RUN donors-probe --api-base http://host:8001 --model donor-grpo-step175 \
    --w 0.75 --q 0.75 --out llm_dynamics/results/probes/trained.json
eval $RUN reciprocity-figure \
    --probes llm_dynamics/results/probes/base.json \
             llm_dynamics/results/probes/trained.json \
    --opponent tit_for_tat

# PD variations at different memories:
eval $RUN matrix-sweep --api-base http://host:8000 --model Qwen/Qwen3-8B \
    --games ipd,chicken,staghunt,harmony --memories 1,2,4 \
    --strategies tit_for_tat,always_defect --rounds 40
eval $RUN matrix-probe --api-base http://host:8000 --model Qwen/Qwen3-8B \
    --game chicken --memory 2
```

Without `--api-base` a scripted `MockLLMClient` is used (`--mock
base-selfish`, `trained-reciprocal`, `soft-tft`, `wsls`, `grim-soft`,
...). The mock parses the real prompts, so it exercises the identical
code path as a served model — use it to validate an experimental design
before spending GPU time.

Serving the checkpoints: **always via SLURM, never directly on rnn.**
`llm_dynamics/slurm/serve_qwen32b.sbatch` serves base `Qwen/Qwen3-32B`
on one A100-80GB (vllm-env, HF cache `/nas/ucb/marimeireles/cache`) and
writes the endpoint to `llm_dynamics/logs/vllm_endpoint.txt` plus a
`.ready` sentinel; `llm_dynamics/slurm/run_base_suite.sbatch` is a
CPU-only client job that waits for the sentinel and runs the whole base
suite (probes, q x w sweep, matrix sweep, figures). Adapt the model path /
`--served-model-name` for trained checkpoints (merged via donorSim's
`merge_fsdp_lora.py`). The client requests `logprobs/top_logprobs` and
toggles Qwen3 thinking via `chat_template_kwargs` (auto-fallback for
servers that reject it). Everything here is stdlib + this repo — no
litellm/torch needed on the client side.

## Files

| file | contents |
|---|---|
| `strategies.py` | vendored donorSim game logic (opponents, payoffs, HKB, parsing, prompt fragments, Nowak thresholds) |
| `llm_client.py` | `VLLMChatClient` (+ logprob `p_cooperate`), `MockLLMClient` |
| `donors_crld.py` | payoff reduction, q-observability, w→γ, memory-m envs, pinned strategies, fixed-opponent flow/trajectory |
| `donors_game.py` | conversational dyadic donors harness (training-faithful) |
| `matrix_games.py` | stateless memory-m PD/Chicken/StagHunt/Harmony harness |
| `policy_probe.py` | P(C \| history state) probes, both framings |
| `plots.py` | cooperation portraits, reciprocity portraits, sweep heatmaps |
| `run_experiments.py` | CLI (see `--help`) |
| `analysis.py` | best-response regret: oracle optimum vs each fixed strategy (DP over its state machine, per partner segment) vs the model’s captured payoff; `python -m llm_dynamics.analysis results/<sweep_dir>` |
| `results/demo_*` | output of the offline `demo` command |

## Caveats

* The two-agent flow background treats *both* seats as CRLD learners
  (repo convention); the fixed-opponent flow in the reciprocity plane is
  the theoretically matched background for the set-strategy scenario.
* `w=1` is clamped to `γ=0.99` (the `(1−γ)` prefactor vanishes at 1).
* `grim_trigger` is pinned as its memory-m truncation;
  `tit_for_two_tats` needs `memory ≥ 2`.
* Probes use forced assistant turns (`DECISION: X` only) in the donors
  framing — a model that writes long rationales sees slightly different
  transcripts during real play.
