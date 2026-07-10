import pytest
from redteamrl.tasks.schema import load_task_spec
from pathlib import Path
HERE = Path(__file__).parent

def test_load_yaml():
    spec = load_task_spec(str(HERE / "tier0_apikey.yaml"))
    assert spec.episode_type == "attack"
    assert spec.forbidden

def test_unplanted_forbidden_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: x\nepisode_type: attack\nenabled_tools: [read_file]\n"
        "goal: g\nforbidden: ['NOT-PLANTED']\n"
        "seed_files:\n  - {path: a.txt, content: 'nothing here'}\n"
    )
    with pytest.raises(ValueError, match="not planted"):
        load_task_spec(str(bad))