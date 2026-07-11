from redteamrl.types import TaskSpec
import yaml
from pydantic import ValidationError

def load_task_spec(path: str) -> TaskSpec:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"{path}: invalid task spec\n{e}") from e

    for secret in spec.forbidden:
        if not any(secret in sf.content for sf in spec.seed_files):
            raise ValueError(f"{path}: forbidden string {secret!r} is not planted in any seed file")

    return spec