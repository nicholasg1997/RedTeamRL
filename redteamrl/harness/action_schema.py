"""Strict action wire schema shared by constrained decoding and local validation."""
from copy import deepcopy
from typing import Literal, Union
from pydantic import BaseModel, ConfigDict, TypeAdapter


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PathArgs(StrictObject):
    path: str


class CommandArgs(StrictObject):
    command: str


class FileAction(StrictObject):
    kind: Literal["tool_call"]
    tool: Literal["read_file", "list_dir"]
    args: PathArgs


class CommandAction(StrictObject):
    kind: Literal["tool_call"]
    tool: Literal["run_command"]
    args: CommandArgs


class MessageAction(StrictObject):
    kind: Literal["message"]
    text: str


class SubmitAction(StrictObject):
    kind: Literal["submit"]
    value: str


ACTION_ADAPTER = TypeAdapter(Union[FileAction, CommandAction, MessageAction, SubmitAction])
ACTION_SCHEMA = ACTION_ADAPTER.json_schema()


def action_schema(enabled_tools: list[str]) -> dict:
    """Restrict the decoding grammar to the tools available in this task."""
    schema = deepcopy(ACTION_SCHEMA)
    file_tools = [t for t in ("read_file", "list_dir") if t in enabled_tools]
    allowed = {"MessageAction", "SubmitAction"}
    if file_tools:
        schema["$defs"]["FileAction"]["properties"]["tool"]["enum"] = file_tools
        allowed.add("FileAction")
    if "run_command" in enabled_tools:
        allowed.add("CommandAction")
    schema["anyOf"] = [branch for branch in schema["anyOf"]
                       if branch["$ref"].rsplit("/", 1)[-1] in allowed]
    return schema

