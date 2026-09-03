import json
import os

from redteamrl.train.capture import Example
from redteamrl.train.episode_store import (
    banked_episode_ids, read_episodes, write_episode,
)


def _payload(episode_id, task_id, reward):
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "reward": reward,
        "tally": {"def_dec": 2, "def_fail": 0, "atk_act": 1, "atk_inv": 0,
                  "ben_act": 0, "ben_inv": 0, "turns": 3},
        "examples": [
            Example([1, 2], [3], task_id=task_id, episode_id=episode_id, reward=reward),
        ],
    }


def test_round_trips_examples_with_full_attribution(tmp_path):
    store = str(tmp_path / "iter0")
    write_episode(store, _payload(7, "task-a", -2.0))

    restored = read_episodes(store)

    assert set(restored) == {7}
    example = restored[7]["examples"][0]
    assert isinstance(example, Example)
    assert (example.prompt_ids, example.completion_ids) == ([1, 2], [3])
    assert example.task_id == "task-a"
    assert example.episode_id == 7
    assert example.reward == -2.0
    assert restored[7]["tally"]["turns"] == 3


def test_only_complete_writes_are_readable(tmp_path):
    # A preemption mid-write must not leave a half-episode that resume would trust. The atomic
    # rename means a partial file never appears under its final name.
    store = str(tmp_path / "iter0")
    write_episode(store, _payload(1, "t", 1.0))
    with open(os.path.join(store, "000002.json.tmp"), "w") as handle:
        handle.write('{"episode_id": 2, "exam')      # torn write

    assert banked_episode_ids(store) == {1}
    assert set(read_episodes(store)) == {1}


def test_corrupt_final_file_is_skipped_not_fatal(tmp_path):
    store = str(tmp_path / "iter0")
    write_episode(store, _payload(1, "t", 1.0))
    with open(os.path.join(store, "000009.json"), "w") as handle:
        handle.write("not json")

    assert set(read_episodes(store)) == {1}


def test_missing_store_is_empty_not_an_error(tmp_path):
    assert read_episodes(str(tmp_path / "nope")) == {}
    assert banked_episode_ids(str(tmp_path / "nope")) == set()
