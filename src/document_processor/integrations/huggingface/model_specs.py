from dataclasses import dataclass

from document_processor.core.registry import Registry
from document_processor.integrations.huggingface.reference import ModelReference


@dataclass(frozen=True, kw_only=True)
class HuggingFaceModelSpec:
    reference: ModelReference
    use_sparse: bool = False

    # Which of the model's declared prompts to use, by name (e.g. "query"), as in
    # sentence-transformers' encode(prompt_name=...). The text comes from the model.
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None

    def __post_init__(self) -> None:
        for name, prompt_name in (
            ("query_prompt_name", self.query_prompt_name),
            ("document_prompt_name", self.document_prompt_name),
        ):
            if prompt_name == "":
                raise ValueError(
                    f"{name} must not be empty; use None to use no prompt"
                )


BGE_M3 = HuggingFaceModelSpec(
    reference=ModelReference(
        hub_model_id="BAAI/bge-m3",
        commit_hash="5617a9f61b028005a4858fdac845db406aefb181",
    ),
    use_sparse=True,
)


HUGGINGFACE_MODEL_SPECS = Registry[HuggingFaceModelSpec](
    kind="embedding model spec", items={"bge-m3": BGE_M3}
)
