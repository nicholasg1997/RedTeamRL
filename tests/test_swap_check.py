"""The publish check compares vLLM to itself: the served adapter vs a fresh load of the same file.

Regression for v3 arm 2 iter 0 (2026-10-18): the old vLLM-vs-HF greedy check raised "serving the
BASE policy" on a swap that worked, because a ~base adapter made the two stacks break a near-tie
differently. A near-base adapter must PASS; a served adapter that is not the one just published
(the real failure) must RAISE.
"""
import pytest

import redteamrl.policies.swap_check as sc

BASE = {"a": -0.1, "b": -2.5, "c": -4.0}
NEW = {"a": -0.11, "b": -2.45, "c": -4.0}      # tiny update: ~base, like iteration 0
OLD = {"a": -0.9, "b": -0.6, "c": -3.0}        # a previous, clearly different adapter


class _FakeVLLM:
    def __init__(self, served_dist):
        # `served_dist` may be a list: successive served requests cycle through it (a noisy engine).
        self.served = served_dist if isinstance(served_dist, list) else [served_dist]
        self.loaded, self.unloaded, self.paths, self.n_served = [], [], {}, 0

    def post(self, url, json=None, timeout=None):
        path = url.split("8001", 1)[1]
        if path == "/v1/load_lora_adapter":
            self.loaded.append(json["lora_name"])
            self.paths[json["lora_name"]] = json["lora_path"]
            return _Resp({})
        if path == "/v1/unload_lora_adapter":
            self.unloaded.append(json["lora_name"])
            return _Resp({})
        model = json["model"]
        if model == "base":
            dist = BASE
        elif model.startswith("swapcheck-"):
            dist = OLD if self.paths[model] == "/tmp/old" else NEW
        else:
            dist = self.served[self.n_served % len(self.served)]
            self.n_served += 1
        return _Resp({"choices": [{"logprobs": {"top_logprobs": [dist]}}]})


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _run(monkeypatch, served_dist):
    fake = _FakeVLLM(served_dist)
    monkeypatch.setattr(sc.requests, "post", fake.post)
    return fake, lambda: sc.verify_served_adapter("http://localhost:8001", "attacker",
                                                  "/tmp/new", "base", ["p1", "p2"])


def test_near_base_adapter_that_loaded_passes(monkeypatch):
    fake, check = _run(monkeypatch, served_dist=NEW)
    d = check()
    assert d["served_vs_fresh"] == pytest.approx(0.0)
    assert d["fresh_vs_base"] < 0.05            # the case the old check got wrong
    assert fake.loaded and fake.unloaded == fake.loaded   # fresh name cleaned up


def test_served_adapter_that_is_not_the_published_one_raises(monkeypatch):
    fake, check = _run(monkeypatch, served_dist=OLD)
    with pytest.raises(RuntimeError, match="does not serve the adapter just published"):
        check()
    assert fake.unloaded == fake.loaded         # cleaned up even on failure


def test_fresh_names_are_never_reused(monkeypatch):
    fake, check = _run(monkeypatch, served_dist=NEW)
    check()
    check()
    assert len(set(fake.loaded)) == 2


def test_stale_swap_is_identified_as_serving_the_previous_adapter(monkeypatch):
    # The real failure: vLLM still serves last iteration's weights. A fresh load of the previous
    # adapter must match the served name exactly, which is what separates it from a noisy check.
    fake = _FakeVLLM(OLD)
    monkeypatch.setattr(sc.requests, "post", fake.post)
    with pytest.raises(sc.SwapMismatch) as err:
        sc.verify_served_adapter("http://localhost:8001", "attacker", "/tmp/new", "base",
                                 ["p1", "p2"], previous_path="/tmp/old")
    m = err.value.measurement
    assert m["served_vs_previous"] == pytest.approx(0.0)
    assert m["served_repeat"] == pytest.approx(0.0)
    assert sorted(fake.unloaded) == sorted(fake.loaded) and len(fake.loaded) == 2


def test_nondeterministic_engine_shows_up_as_a_nonzero_repeat(monkeypatch):
    # The false-alarm signature: the SAME served name disagrees with itself between two passes.
    fake = _FakeVLLM([NEW, NEW, OLD, OLD])          # pass 1 -> NEW, pass 2 -> OLD
    monkeypatch.setattr(sc.requests, "post", fake.post)
    m = sc.measure_served_adapter("http://localhost:8001", "attacker", "/tmp/new", "base",
                                  ["p1", "p2"])
    assert m["served_vs_fresh"] == pytest.approx(0.0)
    assert m["served_repeat"] > 0.05
    assert "served_vs_previous" not in m
    assert m["top1_agree"] == 1.0
