"""Pure (torch-free) checkpoint-directory logic that makes the Modal training run resumable
after a preemption. The torch/peft/optimizer save+load glue lives in the training script and is
verified by reading; THIS module is the branchy part (which dir is 'latest', which are complete,
which to prune) and gets full offline coverage."""
import json
import os

from redteamrl.train.checkpoint import (
    latest_checkpoint, prune_checkpoints, read_meta, write_meta,
)


def _mk(root, it, complete=True):
    """Create a fake checkpoint dir iter{it} with a stand-in adapter file. When complete, the
    meta.json completion marker is written LAST (mirrors the save order in the script)."""
    d = os.path.join(root, f"iter{it}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "adapter_model.safetensors"), "w") as f:
        f.write("weights")
    if complete:
        write_meta(d, {"iter": it})
    return d


def test_latest_is_none_when_root_missing(tmp_path):
    assert latest_checkpoint(str(tmp_path / "does-not-exist")) is None


def test_latest_is_none_when_empty(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None


def test_latest_picks_highest_iter_numerically_not_lexically(tmp_path):
    # iter2 vs iter10: lexical sort would wrongly pick iter2 ("iter10" < "iter2")
    _mk(str(tmp_path), 0)
    _mk(str(tmp_path), 2)
    expected = _mk(str(tmp_path), 10)
    assert latest_checkpoint(str(tmp_path)) == expected


def test_incomplete_checkpoint_is_ignored(tmp_path):
    # a half-written dir (no meta.json) must never be treated as resumable
    _mk(str(tmp_path), 5)
    _mk(str(tmp_path), 6, complete=False)   # newer but incomplete -> skipped
    assert latest_checkpoint(str(tmp_path)) == os.path.join(str(tmp_path), "iter5")


def test_meta_roundtrip(tmp_path):
    d = str(tmp_path)
    write_meta(d, {"iter": 7, "note": "x"})
    assert read_meta(d) == {"iter": 7, "note": "x"}


def test_prune_keeps_newest_k_by_iter(tmp_path):
    for it in range(6):            # iter0..iter5
        _mk(str(tmp_path), it)
    deleted = prune_checkpoints(str(tmp_path), keep=2)

    remaining = sorted(n for n in os.listdir(str(tmp_path)))
    assert remaining == ["iter4", "iter5"]          # newest two kept
    assert len(deleted) == 4                         # iter0..iter3 removed


def test_prune_never_deletes_incomplete_current_write(tmp_path):
    # a fresh, still-being-written dir (no meta) must survive pruning even when it's the newest
    _mk(str(tmp_path), 0)
    _mk(str(tmp_path), 1)
    _mk(str(tmp_path), 2, complete=False)
    prune_checkpoints(str(tmp_path), keep=1)

    remaining = sorted(os.listdir(str(tmp_path)))
    assert "iter2" in remaining                       # incomplete newest untouched
    assert "iter1" in remaining                       # newest COMPLETE kept
    assert "iter0" not in remaining                   # older complete pruned
