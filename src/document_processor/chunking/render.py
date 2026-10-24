from typing import assert_never

from document_processor.core.models import (
    HeadingSegment,
    ImageSegment,
    LogicalDocument,
    Segment,
    TableSegment,
    TextSegment,
    heading_label,
)


def render_segments(document: LogicalDocument, start: int, end: int) -> str:
    if not (0 <= start <= end <= len(document.segments)):
        raise ValueError(
            f"required: 0 <= start <= end <= {len(document.segments)}, got start={start}, end={end}"
        )

    lines: list[str] = []
    previous: Segment | None = None
    for segment in document.segments[start:end]:
        if isinstance(segment, HeadingSegment):
            label = heading_label(segment.marker, segment.title)
            if label:
                lines.append(label)
        elif isinstance(segment, TextSegment):
            if _continues_its_marker(previous, segment):
                lines[-1] = f"{lines[-1]} {segment.text}"
            else:
                lines.append(segment.text)
        elif isinstance(segment, TableSegment):
            lines.append(segment.table.text())
        elif isinstance(segment, ImageSegment):
            # Images have no text
            pass
        else:
            assert_never(segment)
        previous = segment

    return "\n".join(lines)


def _continues_its_marker(previous: Segment | None, segment: TextSegment) -> bool:
    return (
        isinstance(previous, HeadingSegment)
        and previous.title is None
        and previous.marker is not None
        and previous.block_id == segment.block_id
    )