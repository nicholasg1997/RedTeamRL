import os
import threading
import time
from types import SimpleNamespace

import torch

import redteamrl.train.train as train_mod
from redteamrl.train.capture import CapturingGenerate
from redteamrl.train.train import rollout


class _FakeTok:
    def apply_chat_template(self, chat, add_generation_prompt, enable_thinking, return_tensors):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class _FakeModel:
    device = "cpu"

    def generate(self, prompt_ids, attention_mask, do_sample, temperature, max_new_tokens):
        time.sleep(0.01)          # widen the interleaving window
        return torch.cat([prompt_ids, torch.tensor([[4, 5]])], dim=1)


def _cap():
    return CapturingGenerate(_FakeModel(), _FakeTok())


def test_episode_capture_isolates_threads():
    cap = _cap()
    seen = {}

    def worker(name):
        with cap.episode_capture() as buffer:
            cap("sys", [{"role": "user", "content": name}])
            time.sleep(0.02)      # let the other thread capture in between
            cap("sys", [{"role": "user", "content": name}])
            seen[name] = list(buffer)

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each episode sees exactly its own two captures, never the other thread's.
    assert len(seen["a"]) == 2 and len(seen["b"]) == 2
    assert not {id(e) for e in seen["a"]} & {id(e) for e in seen["b"]}


def test_concurrent_rollout_never_cross_attributes_rewards(monkeypatch):
    # Slicing a shared buffer by index interleaves under concurrency: one episode's slice picks up
    # another's examples and tags them with the WRONG reward. That is silent credit-assignment
    # corruption, which is why concurrency was unsafe before per-episode buffers.
    cap = _cap()

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        cap("sys", [{"role": "user", "content": spec.id}])
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(
            defender_reward=1.0 if spec.id == "good" else -2.0,
            attacker_reward=0.0, n_turns=2, steps=[],
            policy_leak=False, complete=True, defender_protocol_failures=0,
        )

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    specs = [SimpleNamespace(id="good", episode_type="benign"),
             SimpleNamespace(id="bad", episode_type="attack")]

    examples = rollout(
        specs, lambda s: None, lambda s: None, lambda: None,
        lambda: SimpleNamespace(close=lambda: None),
        cap, n_rollouts=4, max_workers=8,
    )

    assert len(examples) == 2 * 4 * 2          # 2 tasks x 4 rollouts x 2 captures
    for example in examples:
        expected = 1.0 if example.task_id == "good" else -2.0
        assert example.reward == expected, "example attributed to the wrong episode"
    # Episode ids stay unique and stable regardless of completion order.
    assert len({e.episode_id for e in examples}) == 8


def test_sequential_default_is_unchanged(monkeypatch):
    cap = _cap()

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(defender_reward=1.0, attacker_reward=0.0, n_turns=1, steps=[],
                               policy_leak=False, complete=True, defender_protocol_failures=0)

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    examples = rollout(
        [SimpleNamespace(id="t", episode_type="benign")],
        lambda s: None, lambda s: None, lambda: None,
        lambda: SimpleNamespace(close=lambda: None),
        cap, n_rollouts=3,
    )
    assert [e.episode_id for e in examples] == [0, 1, 2]


def _episode_fake(cap, calls):
    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        calls.append(spec.id)
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(
            defender_reward=1.0 if spec.id == "good" else -2.0,
            attacker_reward=0.0, n_turns=3, steps=[],
            policy_leak=False, complete=True, defender_protocol_failures=0,
        )
    return fake_run_episode


def test_rollout_resumes_from_banked_episodes_instead_of_rerunning(monkeypatch, tmp_path):
    # Checkpointing only between iterations meant a preemption at task 6/9 threw away hours of
    # finished episodes. Banked episodes must be reused, and only the missing ones re-run.
    store = str(tmp_path / "iter3")
    specs = [SimpleNamespace(id="good", episode_type="benign"),
             SimpleNamespace(id="bad", episode_type="attack")]

    first_calls = []
    cap = _cap()
    monkeypatch.setattr(train_mod, "run_episode", _episode_fake(cap, first_calls))
    full = rollout(specs, lambda s: None, lambda s: None, lambda: None,
                   lambda: SimpleNamespace(close=lambda: None),
                   cap, n_rollouts=3, episode_store=store)
    assert len(first_calls) == 6

    # Simulate a preemption after the first task: drop the second task's records.
    for episode_id in (3, 4, 5):
        os.remove(os.path.join(store, f"{episode_id:06d}.json"))

    resumed_calls = []
    cap2 = _cap()
    monkeypatch.setattr(train_mod, "run_episode", _episode_fake(cap2, resumed_calls))
    resumed = rollout(specs, lambda s: None, lambda s: None, lambda: None,
                      lambda: SimpleNamespace(close=lambda: None),
                      cap2, n_rollouts=3, episode_store=store)

    # Only the three missing episodes re-ran; the banked task was not touched.
    assert resumed_calls == ["bad", "bad", "bad"]
    assert len(resumed) == len(full) == 6
    for example in resumed:
        assert example.reward == (1.0 if example.task_id == "good" else -2.0)
    assert {e.episode_id for e in resumed} == {0, 1, 2, 3, 4, 5}


def test_commit_runs_once_per_task_from_the_calling_thread(monkeypatch, tmp_path):
    cap = _cap()
    monkeypatch.setattr(train_mod, "run_episode", _episode_fake(cap, []))
    threads = []
    rollout([SimpleNamespace(id="good", episode_type="benign"),
             SimpleNamespace(id="bad", episode_type="attack")],
            lambda s: None, lambda s: None, lambda: None,
            lambda: SimpleNamespace(close=lambda: None),
            cap, n_rollouts=2, max_workers=4,
            episode_store=str(tmp_path / "iter0"),
            commit=lambda: threads.append(threading.current_thread().name))

    assert len(threads) == 2                      # one per task
    assert set(threads) == {threading.main_thread().name}   # never from a worker
