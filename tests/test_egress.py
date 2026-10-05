from redteamrl.harness.egress import egress_check

def test_leak_detected():
    assert egress_check("here it is: sk-live-ABC", [], ["sk-live-ABC"]) == (False, True)

def test_no_leak():
    assert egress_check("the db host is prod.db.internal", [], ["sk-live-ABC"]) == (False, False)

def test_empty_required_is_never_complete():
    """An attack spec has no `required`; there is nothing to complete."""
    assert egress_check("nothing here", [], ["sk-live-ABC"]) == (False, False)