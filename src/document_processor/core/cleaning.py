from collections.abc import Callable
from dataclasses import dataclass, replace

from document_processor.core.models import BlockType, PhysicalBlock, PhysicalDocument
from document_processor.core.text_utils import is_invisible, normalize_nfc, strip_edges


@dataclass(frozen=True)
class CleaningStep:
    name: str
    apply: Callable[[str], str]


DEFAULT_STEPS = (
    CleaningStep("nfc", normalize_nfc),
    CleaningStep("strip_edges", strip_edges),
)


def _clean_block(
    block: PhysicalBlock, steps: tuple[CleaningStep, ...] = DEFAULT_STEPS
) -> PhysicalBlock:
    if block.block_type == BlockType.TABLE or block.block_type == BlockType.IMAGE:
        return block

    text = block.text
    effective_steps: list[str] = []
    for step in steps:
        new_text = step.apply(text)
        if new_text != text:
            effective_steps.append(f"clean:{step.name}")
        text = new_text

    if not effective_steps:
        return block

    if block.char_spans:
        raise ValueError(
            f"PhysicalBlock {block.id}: not supposed to clean a PhysicalBlock with non-empty char_spans"
        )

    if is_invisible(text):
        raise ValueError(
            f"PhysicalBlock {block.id}: applying {effective_steps} unexpectedly caused the text to become invisible"
        )

    return replace(block, text=text, changelog=block.changelog + tuple(effective_steps))


def clean_document(
    doc: PhysicalDocument, steps: tuple[CleaningStep, ...] = DEFAULT_STEPS
) -> PhysicalDocument:
    return replace(doc, blocks=tuple(_clean_block(b, steps) for b in doc.blocks))
