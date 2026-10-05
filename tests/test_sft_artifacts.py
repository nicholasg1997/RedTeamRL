from redteamrl.sft.artifacts import (
    episode_manifest_identity,
    evaluation_artifact_tag,
    rationalization_artifact_tag,
    rationalization_manifest_identity,
    training_artifact_tag,
    training_manifest_identity,
)


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


def test_balance_change_invalidates_training_but_reuses_rationalizations():
    prior = {
        "pipeline_revision": 9,
        "n_collect": 20,
        "redact_to_pass_ratio": 1.0,
        "sft_lr": 5e-6,
    }
    current = {
        **prior,
        "redact_to_pass_ratio": 2.0,
        "sft_lr": 1e-6,
    }

    assert rationalization_manifest_identity(prior) == rationalization_manifest_identity(current)
    assert training_manifest_identity(prior) != training_manifest_identity(current)
    assert training_artifact_tag(prior) != training_artifact_tag(current)


def test_gate_objective_migration_is_eval_only_for_legacy_manifest():
    prior = {
        "pipeline_revision": 9,
        "n_collect": 20,
        "min_benign_leak_improvement": 0.10,
    }
    current = {
        "pipeline_revision": 9,
        "n_collect": 20,
        "min_attack_policy_improvement": 0.05,
    }

    assert rationalization_manifest_identity(prior) == rationalization_manifest_identity(current)
    assert training_manifest_identity(prior) == training_manifest_identity(current)


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


def test_rationalization_revision_invalidates_authored_work_but_not_episodes():
    # Banked rationalized/*.json record the FILTER's accept/reject verdict. A training-revision
    # bump preserves them by design, so a filter change needs its own revision or the relaxed
    # filter silently never runs.
    from redteamrl.sft.artifacts import (
        rationalization_manifest_identity, training_manifest_identity)

    base = {"schema_version": 2, "n_collect": 20, "training_revision": 1,
            "rationalization_revision": 1, "eval_rollouts": 20}
    filter_changed = {**base, "rationalization_revision": 2}
    recipe_changed = {**base, "training_revision": 2}

    assert rationalization_manifest_identity(base) != rationalization_manifest_identity(filter_changed)
    assert rationalization_artifact_tag(base) != rationalization_artifact_tag(filter_changed)
    assert episode_manifest_identity(base) == episode_manifest_identity(filter_changed)
    assert rationalization_manifest_identity(base) == rationalization_manifest_identity(recipe_changed)
    assert training_manifest_identity(base) != training_manifest_identity(filter_changed)


def test_review_split_change_selects_new_rationalizations_but_preserves_episodes():
    base = {
        "schema_version": 2,
        "pipeline_revision": 9,
        "n_collect": 20,
        "review_holdout_fraction": 0.2,
        "review_split_version": 1,
    }
    changed = {
        **base,
        "review_holdout_fraction": 0.3,
        "review_split_version": 2,
    }

    assert episode_manifest_identity(base) == episode_manifest_identity(changed)
    assert rationalization_manifest_identity(base) != rationalization_manifest_identity(changed)
    assert rationalization_artifact_tag(base) != rationalization_artifact_tag(changed)


def test_collection_change_still_invalidates_episode_bank():
    base = {"pipeline_revision": 9, "n_collect": 20, "rationalization_revision": 1}
    changed = {**base, "n_collect": 40}

    assert episode_manifest_identity(base) != episode_manifest_identity(changed)
