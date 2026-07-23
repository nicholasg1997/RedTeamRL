from redteamrl.sft.artifacts import evaluation_artifact_tag, training_manifest_identity


def test_training_manifest_ignores_eval_only_changes():
    prior = {
        "schema_version": 2,
        "pipeline_revision": 5,
        "n_collect": 20,
        "evaluation_backend": "transformers",
        "eval_rollouts": 20,
        "eval_defender_temperature": 0.0,
    }
    current = {
        **prior,
        "evaluation_backend": "vllm-merged",
        "eval_rollouts": 40,
        "train_eval_rollouts": 4,
        "eval_checkpoint_every": 10,
    }

    assert training_manifest_identity(prior) == training_manifest_identity(current)


def test_training_manifest_keeps_data_changes():
    prior = {"pipeline_revision": 5, "n_collect": 20, "eval_rollouts": 20}
    current = {**prior, "n_collect": 40}

    assert training_manifest_identity(prior) != training_manifest_identity(current)


def test_eval_artifact_tag_is_stable_and_config_specific():
    a = evaluation_artifact_tag("before-held", {"backend": "vllm", "n_rollouts": 20})
    reordered = evaluation_artifact_tag(
        "before-held", {"n_rollouts": 20, "backend": "vllm"}
    )
    changed = evaluation_artifact_tag(
        "before-held", {"backend": "vllm", "n_rollouts": 40}
    )

    assert a == reordered
    assert a != changed
