from redteamrl.types import AttackerAction
from redteamrl.policies.scripted import ScriptedClient
from redteamrl.policies.base import AttackerPolicy

def test_replays_actions_then_done():
    actions = [
        AttackerAction(kind="tool_call", tool="list_dir", args={"path": "."}),
        AttackerAction(kind="message", text="hi"),
    ]
    c = ScriptedClient(actions)
    assert isinstance(c, AttackerPolicy)
    assert c.act([]).tool == "list_dir"
    assert c._i == 1
    assert c.act([]).text == "hi"
    assert c.act([]).text == "(done)"