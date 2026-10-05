import pytest

from redteamrl.train.plasticity import effective_rank, forgetting, stability


def test_effective_rank_of_rank_one_matrix_is_one():
    assert effective_rank([[1.0, 2.0], [2.0, 4.0]]) == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_of_identity_is_full_rank():
    assert effective_rank([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) == pytest.approx(3.0, abs=1e-4)


def test_effective_rank_of_zero_matrix_is_zero():
    assert effective_rank([[0.0, 0.0], [0.0, 0.0]]) == 0.0


def test_effective_rank_collapses_toward_one_when_one_direction_dominates():
    # the plasticity-collapse signature: almost all spectral energy in a single direction
    assert 1.0 <= effective_rank([[1.0, 0.0], [0.0, 1e-4]]) < 1.5


def test_effective_rank_rejects_non_2d_input():
    with pytest.raises(ValueError):
        effective_rank([1.0, 2.0, 3.0])


def test_stability_reports_mean_and_variance_per_task():
    out = stability({"saturated": [1.0, 1.0, 1.0], "mixed": [1.0, -1.0]})
    assert out["saturated"]["mean"] == pytest.approx(1.0)
    assert out["saturated"]["var"] == pytest.approx(0.0)     # converged; read WITH the mean
    assert out["mixed"]["mean"] == pytest.approx(0.0)
    assert out["mixed"]["var"] == pytest.approx(1.0)


def test_stability_skips_tasks_with_no_rollouts():
    assert stability({"none": []}) == {}


def test_forgetting_is_positive_when_a_task_regresses_from_its_peak():
    assert forgetting([{"a": 1.0}, {"a": 1.0}, {"a": 0.0}])["a"] == pytest.approx(1.0)


def test_forgetting_is_negative_while_a_task_is_still_improving():
    assert forgetting([{"a": 0.0}, {"a": 1.0}])["a"] == pytest.approx(-1.0)


def test_forgetting_needs_at_least_two_iterations():
    assert forgetting([{"a": 1.0}]) == {}


def test_forgetting_skips_tasks_with_no_prior_history():
    out = forgetting([{"a": 1.0}, {"a": 1.0, "b": 0.5}])
    assert set(out) == {"a"}
    assert out["a"] == pytest.approx(0.0)
