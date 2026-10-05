from redteamrl.types import AttackerAction
from redteamrl.policies.base import DefenderPolicy
from redteamrl.verify.defenders import NullDefender, OracleRedactorDefender

ACT = AttackerAction(kind="tool_call", tool="read_file", args={"path": "c.py"})

def test_null_allows_and_passes():
    d = NullDefender()
    assert isinstance(d, DefenderPolicy)
    assert d.adjudicate_call([], ACT).verdict == "allow"
    assert d.adjudicate_result([], ACT, "x").verdict == "pass"

def test_oracle_scrubs_forbidden_keeps_rest():
    d = OracleRedactorDefender(forbidden=["sk-live-X"])
    dec = d.adjudicate_result([], ACT, 'HOST="prod.db.internal"\nK="sk-live-X"')
    assert dec.verdict == "redact"
    assert "sk-live-X" not in dec.content
    assert "prod.db.internal" in dec.content