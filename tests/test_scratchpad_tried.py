from redteamrl.scratchpad.tried import summarize_action, summarize_response, tried_lines
from redteamrl.types import AttackerAction


def _call(tool, **args):
    return AttackerAction(kind="tool_call", tool=tool, args=args)


def test_line_format_action_arrow_response():
    lines = tried_lines([_call("read_file", path="config/.env")], ["[REDACTED by monitor]"])
    assert lines == ["read_file config/.env -> [REDACTED by monitor]"]


def test_long_response_truncated_with_marker():
    [line] = tried_lines([_call("list_dir", path=".")], ["x" * 500], max_chars=160)
    assert len(line) < 220 and line.endswith("...")


def test_newest_action_without_response_is_omitted():
    lines = tried_lines([_call("list_dir", path="."), _call("read_file", path="a")], ["3 files"])
    assert len(lines) == 1


def test_response_newlines_collapse_to_one_line():
    assert "\n" not in summarize_response("a\nb\n\nc", 160)


def test_message_and_submit_summaries():
    assert summarize_action(AttackerAction(kind="message", text="hi")).startswith("message")
    assert summarize_action(AttackerAction(kind="submit", value="SECRET")).startswith("submit")


def test_run_command_summary_shows_the_command():
    assert summarize_action(_call("run_command", command="ls -la")) == "run_command ls -la"


def test_long_action_detail_is_truncated_so_tried_cannot_grow_unbounded():
    [line] = tried_lines([_call("run_command", command="x" * 5000)], ["ok"], max_chars=160)
    assert len(line) < 400
