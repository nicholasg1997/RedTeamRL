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


def test_episodes_from_different_tasks_run_at_the_same_time(monkeypatch, tmp_path):
    # Tasks used to run one after another, so peak concurrency was capped at n_rollouts (6) no
    # matter how many workers existed. vLLM batches by in-flight request count, so a queue of one
    # leaves the engine at batch-1 -- the whole throughput advantage unused.
    cap = _cap()
    peak = 0
    live = 0
    lock = threading.Lock()

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        cap("sys", [{"role": "user", "content": spec.id}])
        with lock:
            live -= 1
        return SimpleNamespace(defender_reward=1.0, attacker_reward=0.0, n_turns=2, steps=[],
                               policy_leak=False, complete=True, defender_protocol_failures=0)

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    specs = [SimpleNamespace(id=f"t{i}", episode_type="benign") for i in range(4)]

    rollout(specs, lambda s: None, lambda s: None, lambda: None,
            lambda: SimpleNamespace(close=lambda: None),
            cap, n_rollouts=3, max_workers=12)

    assert peak > 3, f"peak concurrency {peak} never exceeded one task's worth of episodes"


def test_examples_stay_in_deterministic_task_order_despite_completion_order(monkeypatch):
    cap = _cap()

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        # Later tasks finish FIRST, so arrival order is the reverse of task order.
        time.sleep(0.02 * (3 - int(spec.id[-1])))
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(defender_reward=1.0, attacker_reward=0.0, n_turns=1, steps=[],
                               policy_leak=False, complete=True, defender_protocol_failures=0)

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    specs = [SimpleNamespace(id=f"t{i}", episode_type="benign") for i in range(3)]

    examples = rollout(specs, lambda s: None, lambda s: None, lambda: None,
                       lambda: SimpleNamespace(close=lambda: None),
                       cap, n_rollouts=2, max_workers=8)

    assert [e.task_id for e in examples] == ["t0", "t0", "t1", "t1", "t2", "t2"]
    assert [e.episode_id for e in examples] == [0, 1, 2, 3, 4, 5]


def test_task_transform_gives_each_rollout_its_own_task_without_breaking_attribution(monkeypatch):
    # SFT randomized the planted canary per episode; GRPO never did, so it ran against the literal
    # template `tok-deploy-CANARY-3m2n1k` -- a low-entropy string that reads like a placeholder.
    # The two phases were measuring different environments.
    cap = _cap()
    seen = []

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        seen.append(spec.forbidden[0])
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(defender_reward=1.0, attacker_reward=0.0, n_turns=1, steps=[],
                               policy_leak=False, complete=True, defender_protocol_failures=0)

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    spec = SimpleNamespace(id="t0", episode_type="benign", forbidden=["TEMPLATE"])

    def transform(original, rollout_index):
        return SimpleNamespace(id=original.id, episode_type=original.episode_type,
                               forbidden=[f"secret-{rollout_index}"])

    examples = rollout([spec], lambda s: None, lambda s: None, lambda: None,
                       lambda: SimpleNamespace(close=lambda: None),
                       cap, n_rollouts=3, task_transform=transform)

    assert sorted(seen) == ["secret-0", "secret-1", "secret-2"]
    # The transform must preserve identity: attribution keys off the task, not the secret.
    assert {e.task_id for e in examples} == {"t0"}
    assert [e.episode_id for e in examples] == [0, 1, 2]


def test_no_transform_leaves_the_task_untouched(monkeypatch):
    cap = _cap()
    seen = []

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                        post_completion_turns=None, redaction_enforcement="unshielded"):
        seen.append(spec.forbidden[0])
        cap("sys", [{"role": "user", "content": spec.id}])
        return SimpleNamespace(defender_reward=1.0, attacker_reward=0.0, n_turns=1, steps=[],
                               policy_leak=False, complete=True, defender_protocol_failures=0)

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    rollout([SimpleNamespace(id="t0", episode_type="benign", forbidden=["TEMPLATE"])],
            lambda s: None, lambda s: None, lambda: None,
            lambda: SimpleNamespace(close=lambda: None), cap, n_rollouts=2)

    assert seen == ["TEMPLATE", "TEMPLATE"]
