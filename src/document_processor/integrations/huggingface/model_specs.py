from dataclasses import dataclass

from document_processor.core.registry import Registry
from document_processor.integrations.huggingface.reference import ModelReference


@dataclass(frozen=True, kw_only=True)
class HuggingFaceModelSpec:
    """Which model, and the choices we make among what it offers; never facts about it."""

    reference: ModelReference

    # Only for models whose prompt names aren't conventional: which declared prompt to use
    # for each role, by name. Left None, each role gets the model's own prompt, or none.
    query_prompt_name: str | None = None
    passage_prompt_name: str | None = None

    def __post_init__(self) -> None:
        for field_name, prompt_name in (
            ("query_prompt_name", self.query_prompt_name),
            ("passage_prompt_name", self.passage_prompt_name),
        ):
            if prompt_name == "":
                raise ValueError(
                    f"{field_name} must not be empty; use None to use no prompt"
                )


BGE_M3 = HuggingFaceModelSpec(
    reference=ModelReference(
        hub_model_id="BAAI/bge-m3",
        commit_hash="5617a9f61b028005a4858fdac845db406aefb181",
    ),
)


HUGGINGFACE_MODEL_SPECS = Registry[HuggingFaceModelSpec](
    kind="embedding model spec", items={"bge-m3": BGE_M3}
)
