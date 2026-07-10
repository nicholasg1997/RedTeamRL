from redteamrl.sandbox.local import LocalSandbox
from redteamrl.types import TaskSpec, SeedFile

def _spec():
    return TaskSpec(
        id="t",
        episode_type="attack",
        enabled_tools=["read_file", "list_dir"],
        goal="g",
        forbidden=["SECRET"],
        seed_files=[
            SeedFile(
                path="app/config.py",
                content="KEY = 'SECRET'"
            ),
            SeedFile(
                path="app/data/info.txt",
                content="normal data"
            ),
        ],
    )

def test_happy_path():
    sb = LocalSandbox()
    sb.reset(_spec())

    assert "SECRET" in sb.read_file("app/config.py")
    assert "config.py" in sb.list_dir("app")
    assert "info.txt" in sb.list_dir("app/data")
    assert "normal data" in sb.read_file("app/data/info.txt")
    sb.close()

def test_cannot_escape_with_parent_directory():
    sb = LocalSandbox()
    sb.reset(_spec())

    result = sb.read_file("../../etc/passwd")

    assert "access denied" in result

    sb.close()


def test_cannot_escape_with_deep_parent_traversal():
    sb = LocalSandbox()
    sb.reset(_spec())

    result = sb.read_file(
        "app/data/../../../../../../etc/passwd"
    )

    assert "access denied" in result

    sb.close()


def test_cannot_access_absolute_paths():
    sb = LocalSandbox()
    sb.reset(_spec())

    result = sb.read_file("/etc/passwd")

    assert "access denied" in result

    sb.close()


def test_cannot_list_directory_outside_sandbox():
    sb = LocalSandbox()
    sb.reset(_spec())

    result = sb.list_dir("../../")

    assert "access denied" in result

    sb.close()


def test_valid_nested_path_still_works():
    sb = LocalSandbox()
    sb.reset(_spec())

    result = sb.read_file("app/data/info.txt")

    assert result == "normal data"

    sb.close()

def test_escape_cannot_read_a_real_file_above_root():
    sb = LocalSandbox(); sb.reset(_spec())
    outside = sb.root.parent / "host_secret_probe.txt"
    outside.write_text("HOST_SECRET_DO_NOT_LEAK")
    try:
        result = sb.read_file("../host_secret_probe.txt")
        assert "HOST_SECRET_DO_NOT_LEAK" not in result
        assert "access denied" in result
    finally:
        outside.unlink(missing_ok=True)
        sb.close()