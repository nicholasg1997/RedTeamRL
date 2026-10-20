"""Metric dicts for the dashboard: exact, prefix-categorised keys built from what the drivers
already compute. Pure functions, so the key names are pinned offline."""
import re

import pytest

ATTACKER_KEYS = {
    "learning.live_frac", "learning.dead_groups", "learning.mixed_reward", "learning.fresh_win",
    "learning.fresh_win_task_avg", "learning.mean_reward",
    "restart.win_rate", "restart.fallbacks", "restart.groups",
    "diversity.win_routes", "diversity.bank_size", "diversity.banked_wins", "diversity.banked_near",
    "diversity.encoded_pass_rate", "diversity.leaks.literal",
    "throughput.rollout_min", "throughput.update_min", "throughput.gen_tokens",
    "throughput.tokens_per_s", "throughput.peak_gib",
    "update.loss", "update.kl", "update.grad_norm", "update.ratio",
}
DEFENDER_KEYS = {
    "learning.live_frac", "learning.dead_groups", "learning.mixed_reward", "learning.mean_reward",
    "defender.attack_leak", "defender.benign_leak", "defender.protocol_fail",
    "defender.over_refusal", "defender.clean",
    "diversity.encoded_pass_rate", "diversity.leaks.literal",
    "throughput.rollout_min", "throughput.update_min", "throughput.gen_tokens",
    "throughput.tokens_per_s", "throughput.peak_gib",
    "update.loss", "update.kl", "update.grad_norm", "update.ratio",
}


def _inputs():
    div = {"dead_group_rate": 0.5, "mixed_reward_group_rate": 0.5}
    rew = {"mean_reward": 0.08}
    summary = {"fresh_win_rate": 0.1, "fresh_win_rate_task_avg": 0.12, "restart_win_rate": 0.29,
               "fallbacks": 0, "n_restart_groups": 6, "banked_wins": 17, "banked_near": 12}
    tactics = {"leaks": {"literal": 3}, "exposed": {"literal": 3}, "encoded_pass_rate": 0.0625,
               "n_episodes": 96}
    m = {"loss": -0.0001, "mean_kl": 0.0028, "grad_norm": 0.025, "mean_ratio": 1.001}
    decomp = {"attack_policy_leak": 0.2, "benign_policy_leak": 0.05, "protocol_failure": 0.01,
              "refuse": 0.03, "clean": 0.7}
    return div, rew, summary, tactics, m, decomp


def _attacker(heldout=None, rollout_min=5.0, gen_tokens=72000, n_examples=900):
    from redteamrl.train.iter_metrics import attacker_metrics
    div, rew, summary, tactics, m, _ = _inputs()
    return attacker_metrics(live=386, n_examples=n_examples, div=div, rew=rew, summary=summary,
                            tactics=tactics, update=m, rollout_min=rollout_min, update_min=9.0,
                            gen_tokens=gen_tokens, peak_gib=26.7, banks_size=127, win_routes=29,
                            heldout=heldout)


def _defender(rollout_min=5.0, gen_tokens=72000):
    from redteamrl.train.iter_metrics import defender_metrics
    div, rew, _, tactics, m, decomp = _inputs()
    return defender_metrics(live=300, n_examples=900, div=div, rew=rew, decomp=decomp,
                            tactics=tactics, update=m, rollout_min=rollout_min, update_min=9.0,
                            gen_tokens=gen_tokens, peak_gib=20.0)


def test_attacker_keys_are_exact_and_prefixed():
    got = _attacker()
    assert set(got) == ATTACKER_KEYS
    assert got["diversity.leaks.literal"] == 3
    assert all(re.match(r"^(learning|restart|diversity|throughput|update)\.", k) for k in got)
    assert all(isinstance(v, (int, float)) or v is None for v in got.values())


def test_heldout_keys_absent_not_zero_without_heldout():
    assert not any(k.startswith("heldout.") for k in _attacker())
    got = _attacker(heldout={"win_rate": 0.125, "stalled_rate": 0.3, "dead_rate": 0.05})
    assert got["heldout.win_rate"] == 0.125
    assert got["heldout.stalled_rate"] == 0.3 and got["heldout.dead_rate"] == 0.05
    assert set(got) - {"heldout.win_rate", "heldout.stalled_rate", "heldout.dead_rate"} == ATTACKER_KEYS


def test_defender_keys_are_exact():
    got = _defender()
    assert set(got) == DEFENDER_KEYS
    assert got["defender.over_refusal"] == _inputs()[5]["refuse"]


def test_values_map_from_source_keys():
    got = _attacker()
    assert got["learning.dead_groups"] == 0.5
    assert got["learning.live_frac"] == pytest.approx(386 / 900)
    assert got["restart.win_rate"] == 0.29 and got["restart.groups"] == 6
    assert got["update.kl"] == 0.0028 and got["update.ratio"] == 1.001
    assert got["diversity.win_routes"] == 29 and got["diversity.bank_size"] == 127


def test_tokens_per_s_none_when_rollout_min_zero():
    assert _attacker(rollout_min=0.0)["throughput.tokens_per_s"] is None
    assert _defender(rollout_min=0.0)["throughput.tokens_per_s"] is None
    assert _attacker(rollout_min=2.0, gen_tokens=1200)["throughput.tokens_per_s"] == pytest.approx(10.0)
    assert _attacker(n_examples=0)["learning.live_frac"] is None


def _ex(episode_id, attribution=None, taint=None):
    from redteamrl.train.capture import Example
    return Example(prompt_ids=[1], completion_ids=[2], episode_id=episode_id,
                   leak_attribution=attribution, passed_taint_counts=taint or {})


def _fixture():
    return [_ex(0, "base64", {"base64": 1}), _ex(0, "base64", {"base64": 1}),
            _ex(1, "unattributed"), _ex(2, None, {"fragment": 2}), _ex(3),
            _ex(4, "base64+hex", {"hex": 1, "base64": 1})]


def test_tactic_counts_per_episode():
    from redteamrl.train.train import tactic_counts
    assert tactic_counts(_fixture()) == {
        "leaks": {"base64": 1, "base64+hex": 1, "unattributed": 1},
        "exposed": {"base64": 2, "fragment": 1, "hex": 1},
        "encoded_pass_rate": 0.4, "n_episodes": 5}


def test_tactic_summary_output_unchanged():
    from redteamrl.train.train import tactic_summary
    assert tactic_summary(_fixture()) == (
        "leaks by tactic: base64=1 base64+hex=1 unattributed=1 | "
        "episodes passing a carrier: base64=2 fragment=1 hex=1 | encoded_pass_rate=40.0%")
    assert tactic_summary([]) == (
        "leaks by tactic: none | episodes passing a carrier: none | encoded_pass_rate=0.0%")


def _v6_summary(**overrides):
    summary = dict(_inputs()[2], n_base_groups=12, dead_before=9, dead_after=6, replacements=4,
                   replacements_live=3, depth_frac_mean=0.45, win_by_depth={"lo": 0.0, "hi": 0.5},
                   backed_off={"gen-6": 2})
    summary.update(overrides)
    return summary


def _attacker_with(summary):
    from redteamrl.train.iter_metrics import attacker_metrics
    div, rew, _, tactics, m, _ = _inputs()
    return attacker_metrics(live=386, n_examples=900, div=div, rew=rew, summary=summary,
                            tactics=tactics, update=m, rollout_min=5.0, update_min=9.0,
                            gen_tokens=72000, peak_gib=26.7, banks_size=127, win_routes=29)


def test_v6_restart_keys_present_only_with_v6_summary():
    assert "restart.dead_before" not in _attacker()
    got = _attacker_with(_v6_summary())
    assert got["restart.dead_before"] == 0.75 and got["restart.dead_after"] == 0.5
    assert got["restart.replacements"] == 4 and got["restart.replacements_live"] == 3
    assert got["restart.win_by_depth.lo"] == 0.0 and got["restart.win_by_depth.hi"] == 0.5
    assert "restart.win_by_depth.mid" not in got
    assert got["restart.backed_off_tasks"] == 1
    assert all(isinstance(v, (int, float)) or v is None for v in got.values())


def test_v6_rates_none_when_no_base_groups():
    assert _attacker_with(_v6_summary(n_base_groups=0))["restart.dead_before"] is None


# --- v7 -------------------------------------------------------------------------------------------

V7_UPDATE = {"steps": 16, "early_stopped": False, "kl_iter": 0.008, "clip_frac": 0.02, "lr": 4.5e-5}


def test_v7_update_metrics_present_when_reported():
    from redteamrl.train.iter_metrics import attacker_metrics
    div, rew, summary, tactics, m, _ = _inputs()
    got = attacker_metrics(live=386, n_examples=900, div=div, rew=rew, summary=summary,
                           tactics=tactics, update={**m, **V7_UPDATE}, rollout_min=5.0,
                           update_min=9.0, gen_tokens=72000, peak_gib=26.7, banks_size=127,
                           win_routes=29)
    assert got["update.steps"] == 16 and got["update.early_stopped"] == 0
    assert got["update.kl_iter"] == 0.008 and got["update.clip_frac"] == 0.02
    assert got["update.lr"] == 4.5e-5


def test_defender_update_without_new_keys_still_builds():
    assert set(_defender()) == DEFENDER_KEYS


def test_v7_restart_metrics_present_only_with_graduation_summary():
    from redteamrl.train.iter_metrics import attacker_metrics
    div, rew, summary, tactics, m, _ = _inputs()
    v7 = {**summary, "graduated": 2, "graduated_total": 5, "same_iter_replacements": 3,
          "same_iter_replacements_live": 1}
    got = attacker_metrics(live=386, n_examples=900, div=div, rew=rew, summary=v7,
                           tactics=tactics, update=m, rollout_min=5.0, update_min=9.0,
                           gen_tokens=72000, peak_gib=26.7, banks_size=127, win_routes=29)
    assert (got["restart.graduated"], got["restart.graduated_total"]) == (2, 5)
    assert (got["restart.same_iter_replacements"], got["restart.same_iter_replacements_live"]) == (3, 1)
    assert "restart.graduated" not in _attacker()
