from redteamrl.train.capture import Example
from redteamrl.train.train import assign_advantages


def test_assign_advantages_broadcasts_episode_reward():
    # task T, 2 episodes: ep0 reward +1 (2 decisions), ep1 reward -2 (1 decision)
    exs = [
        Example([1], [2], task_id="T", episode_id=0, reward=1.0),
        Example([1], [3], task_id="T", episode_id=0, reward=1.0),
        Example([1], [4], task_id="T", episode_id=1, reward=-2.0),
    ]
    assign_advantages(exs)
    # both decisions of ep0 share one advantage; ep1's is the opposite sign
    assert exs[0].advantage == exs[1].advantage
    assert exs[0].advantage > 0 > exs[2].advantage


def test_assign_advantages_dead_task_is_zero():
    exs = [
        Example([1], [2], task_id="T", episode_id=0, reward=1.0),
        Example([1], [3], task_id="T", episode_id=1, reward=1.0),
    ]
    assign_advantages(exs)
    assert all(e.advantage == 0.0 for e in exs)