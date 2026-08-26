"""EigenBench run spec: donorSim GRPO adapter vs. its un-RLed base.

Panel design
------------
Four models, three families. Both arms of our experiment are in the panel, and
-- as EigenBench requires -- every model is also a judge.

    Qwen3-32B base        our control arm
    donorSim step_175     our treatment arm (LoRA over the SAME base weights)
    Gemma 3 27B           external judge, family 2
    Mistral Small 3.2     external judge, family 3

Two arms alone cannot be run: the Bradley-Terry-Davison fit collapses to a
single Elo difference and a 2x2 EigenTrust matrix has no consensus to extract.
The externals exist to give the arms something to be ranked against.

We deliberately avoid two models from one family. EigenTrust reads inter-judge
agreement as reliability, so a correlated pair (two Gemmas, say) becomes a
voting bloc that dominates the consensus.

Known bias to report around: Qwen3-32B judges its own LoRA descendant. Refit the
BTD with Qwen excluded as a judge and check the ranking survives -- the
per-judge comparisons are all kept in evaluations.jsonl, so this needs no re-run.

Scenarios
---------
AIRiskDilemmas, 400 dilemmas stratified across its 9 risk categories (see
build_scenarios.py). Chosen over AskReddit/OpenAssistant because behaving well
has to *cost* something for the measurement to have any dynamic range -- a
kindness rubric applied to open chit-chat scores both arms alike and measures
noise.

Constitution
------------
This file runs ONE constitution; run_battery.sh sweeps the full set. The battery
is what makes a result interpretable:

    positive     kindness, oct_goodness       expect RL higher
    reverse      oct_misalignment,            expect RL LOWER (these reward
                 oct_sycophancy               taking pleasure in harm, flattery)
    discriminant oct_humor, oct_poeticism     expect NO movement

If the RL arm rises on every axis including humour, that is judges liking it
more, not ethics. The flat axes are the control.
"""

RUN_SPEC = {
    "verbose": False,
    "models": {
        "Qwen3-32B base": {
            "provider": "hf_local", "kind": "base", "repo_id": "Qwen/Qwen3-32B",
        },
        "donorSim step_175": {
            "provider": "hf_local", "kind": "lora",
            "repo_id": "3l3ktr4/qwen3-32b-donorsim-loras",
            "subfolder": "step_175",
            # Overrides the adapter's stale base path. The shipped
            # adapter_config.json still points at the training node's
            # /dev/shm/verl-cache/... which does not exist anywhere else.
            "base_model_id": "Qwen/Qwen3-32B",
        },
        # Gemma 3, not Gemma 4: the container's vLLM 0.27.1 cannot parse Gemma 4's
        # heterogeneous per-layer config and dies with AmbiguousGlobalPerLayerAttributeError
        # on `head_dim`. Gemma 3 27B is dense-headed (head_dim=128 across all 62 layers)
        # and loads cleanly. Same family, so the panel's three-family design is intact.
        "Gemma 3 27B": {
            "provider": "hf_local", "kind": "base", "repo_id": "google/gemma-3-27b-it",
        },
        "Mistral Small 3.2 24B": {
            "provider": "hf_local", "kind": "base",
            "repo_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        },
    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 0,
        "count": 400,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,          # kindness ships exactly 8
    },
    "collection": {
        "enabled": True,
        "cached_responses_path": None,
        "allow_ties": True,
        # With only 4 models, exhaustive is affordable and strictly better than
        # sampling: 400 scenarios x 4 judges, and every arm is scored by every
        # judge on every scenario, so no arm gets a lucky judge draw.
        "sampler_mode": "all_to_all",
        "group_size": 4,
        "groups": 1,
        "alpha": 2.0,
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
}
