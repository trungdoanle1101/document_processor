from collections.abc import Mapping
from typing import Literal

type PromptRole = Literal["query", "document"]

# Looked for in this order, as sentence-transformers' encode_query and encode_document do.
CONVENTIONAL_PROMPT_NAMES: Mapping[PromptRole, tuple[str, ...]] = {
    "query": ("query",),
    "document": ("document", "passage", "corpus"),
}
_ALL_CONVENTIONAL_NAMES = frozenset(
    name for names in CONVENTIONAL_PROMPT_NAMES.values() for name in names
)


def declared_prompts(prompts: Mapping[str, str | None]) -> dict[str, str]:
    """The model's prompts without empty ones: an empty prompt changes nothing."""
    return {name: text for name, text in prompts.items() if text}


def resolve_prompt_names(
    prompts: Mapping[str, str | None], *, query: str | None, document: str | None
) -> tuple[str | None, str | None]:
    """The prompt names to use for queries and for documents; None means no prompt.

    prompts is the model's own prompts dictionary, empty ones included.
    """
    declared = declared_prompts(prompts)
    unconventional = sorted(declared.keys() - _ALL_CONVENTIONAL_NAMES)
    if unconventional:
        unnamed = [
            field
            for field, chosen in (
                ("query_prompt_name", query),
                ("document_prompt_name", document),
            )
            if chosen is None
        ]
        if unnamed:
            raise ValueError(
                f"the model declares prompts with names the convention does not know ({', '.join(unconventional)}; "
                f"all: {', '.join(sorted(declared))}); name the ones to use in the spec: {' and '.join(unnamed)}"
            )
    return resolve_prompt_name(prompts, "query", query), resolve_prompt_name(
        prompts, "document", document
    )


def resolve_prompt_name(
    prompts: Mapping[str, str | None], role: PromptRole, chosen: str | None
) -> str | None:
    """The prompt to use for one role: the chosen one, else the model's own for that role, else None."""
    declared = declared_prompts(prompts)
    if chosen is not None:
        if chosen not in prompts:
            known = ", ".join(sorted(prompts)) or "none"
            raise ValueError(
                f"{role} prompt {chosen!r} is not one the model has; it has: {known}"
            )
        return chosen if chosen in declared else None  # an empty prompt is no prompt
    for name in CONVENTIONAL_PROMPT_NAMES[role]:
        if name in declared:
            return name
    return None
