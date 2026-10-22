from typing import assert_never

from document_processor.core.models import (
    HeadingSegment,
    ImageSegment,
    LogicalDocument,
    TableSegment,
    TextSegment,
    heading_label,
)


def render_segments(document: LogicalDocument, start: int, end: int) -> str:
    if not (0 <= start <= end <= len(document.segments)):
        raise ValueError(
            f"required: 0 <= start <= end <= {len(document.segments)}, got start={start}, end={end}"
        )

    builder: list[str] = []
    for segment in document.segments[start:end]:
        if isinstance(segment, HeadingSegment):
            text = heading_label(segment.marker, segment.title)
            if text:
                builder.append(text)
        elif isinstance(segment, TextSegment):
            builder.append(segment.text)
        elif isinstance(segment, TableSegment):
            builder.append(segment.table.text())
        elif isinstance(segment, ImageSegment):
            # Images have no text
            continue
        else:
            assert_never(segment)

    return "\n".join(builder)
