from redteamrl.sft.review_split import is_review_held_out, split_episode_keys


def _keys(n=100):
    return [f"task-{i % 4}#{i}" for i in range(n)]


def test_split_is_deterministic_and_content_addressed():
    a = split_episode_keys(_keys(), fraction=0.2, seed=7)
    b = split_episode_keys(_keys(), fraction=0.2, seed=7)
    assert a == b
    assert split_episode_keys(_keys(), fraction=0.2, seed=8) != a


def test_held_out_share_is_approximately_the_requested_fraction():
    train, review = split_episode_keys(_keys(200), fraction=0.2, seed=0)
    assert len(train) + len(review) == 200
    assert 0.15 <= len(review) / 200 <= 0.25


def test_partitions_are_disjoint():
    train, review = split_episode_keys(_keys(200), fraction=0.2, seed=0)
    assert set(train).isdisjoint(review)


def test_membership_does_not_depend_on_the_rest_of_the_corpus():
    # A resumed run may enumerate a different number of banked episodes. If membership shifted
    # with corpus size, the "held-out" set would silently absorb episodes already trained on.
    small = split_episode_keys(_keys(40), fraction=0.2, seed=3)[1]
    large = split_episode_keys(_keys(200), fraction=0.2, seed=3)[1]
    assert set(small) <= set(large)
    assert all(is_review_held_out(key, fraction=0.2, seed=3) for key in large)


def test_every_task_can_contribute_to_both_sides():
    train, review = split_episode_keys(_keys(200), fraction=0.2, seed=0)
    assert {key.split("#")[0] for key in review} == {key.split("#")[0] for key in train}
