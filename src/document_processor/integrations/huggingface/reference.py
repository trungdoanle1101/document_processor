import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_REFERENCE_FILE = "model_reference.json"


@dataclass(frozen=True, kw_only=True)
class ModelReference:
    hub_model_id: str
    commit_hash: str

    def __post_init__(self) -> None:
        if not self.hub_model_id:
            raise ValueError("hub_model_id must not be empty")

        if not re.fullmatch(r"[0-9a-f]{40}", self.commit_hash):
            raise ValueError(
                f"commit_hash must be 40-character lower-case hex, got {self.commit_hash!r}"
            )


def read_model_reference(folder: Path | str) -> ModelReference:
    """Read model reference from folder"""
    path = Path(folder) / MODEL_REFERENCE_FILE
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return ModelReference(
                hub_model_id=data["hub_model_id"], commit_hash=data["commit_hash"]
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"cannot find {MODEL_REFERENCE_FILE} in {folder}; download the model using download_model(), or copy the whole folder from a machine that has it"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{path} is malformed: needs hub_model_id and commit_hash"
        ) from e


def write_model_reference(folder: Path, reference: ModelReference) -> None:
    path = folder / MODEL_REFERENCE_FILE
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(reference), f, indent=2)
        f.write("\n")
