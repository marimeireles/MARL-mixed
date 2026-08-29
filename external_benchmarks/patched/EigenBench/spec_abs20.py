"""EigenBench run spec for the donorSim RL arm (Qwen3-8B, mixed Mode A/B, step 6) (responses only; judging is done
externally, see donorsim_pipeline/eigenbench/README.md).
Add the trained arm as a second hf_local model (kind="lora", subfolder=...)
or a merged base model once it exists."""
RUN_SPEC = {
    "verbose": False,
    "models": {
        "Qwen3-8B donorsim abstract step20": {
            # merged bf16 weights (LoRA folded in), abstract step 20 of the same run
            "provider": "hf_local", "kind": "base",
            "repo_id": "3l3ktr4/donorsim-qwen3-8b-abstract-step20",
        },
    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 0,
        "count": 10398,
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
