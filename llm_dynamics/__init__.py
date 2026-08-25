"""llm_dynamics: cooperation-dynamics experiments for LLM agents.

Applies the MARL-mixed methodology (CRLD flow fields + measured learning
trajectories, cf. jaxmarl_env/algo_phase.py) to the donorSim donors game
and to classic matrix games (PD, Chicken, Stag Hunt, Harmony) played by an
LLM against fixed strategies.

Modules:
  strategies    - fixed opponent strategies, payoffs, HKB phase, decision
                  parsing (vendored from donorSim@neurips-methodology)
  llm_client    - OpenAI-compatible chat client with logprob p(cooperate)
                  readout, plus a scripted MockLLMClient for offline runs
  donors_crld   - the donors game as a CRLD environment: (b,c) payoff
                  reduction, w -> discount factor, q -> observability blend,
                  memory-m embedding, fixed-opponent flow fields
  donors_game   - dyadic donors-game harness (LLM vs set strategy),
                  prompts mirroring the verl/GRPO training rollouts
  matrix_games  - stateless memory-m matrix-game harness (PD variants)
  policy_probe  - measure the LLM's conditional policy P(C | history state)
  plots         - phase portraits: CRLD flow background + LLM trajectories
  run_experiments - CLI entry point
"""
