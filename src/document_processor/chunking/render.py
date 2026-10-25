from typing import Literal, assert_never

from document_processor.core.models import (
    HeadingSegment,
    ImageSegment,
    LogicalDocument,
    Segment,
    TableCell,
    TableInfo,
    TableSegment,
    TextSegment,
    heading_label,
)

type TableLayout = Literal["grid", "key_value"]


def render_table_rows(table: TableInfo, start: int, end: int, layout: TableLayout) -> str:
    """Render the header rows, then body rows [start, end). Rows count from 0, as cell.row does."""
    if not (table.n_header_rows <= start <= end <= table.n_rows):
        raise ValueError(f"required: {table.n_header_rows} <= start <= end <= {table.n_rows}, got start={start}, end={end}")

    if layout == "key_value":
        return _render_table_rows_as_key_value(table, start, end)
    return _render_table_rows_as_grid(table, start, end)


def render_segments(document: LogicalDocument, start: int, end: int, table_layout: TableLayout) -> str:
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
            table = segment.table
            lines.append(render_table_rows(table, table.n_header_rows, table.n_rows, table_layout))
        elif isinstance(segment, ImageSegment):
            # Images have no text
            pass
        else:
            assert_never(segment)
        previous = segment

    return "\n".join(lines)


def _render_table_rows_as_grid(table: TableInfo, start: int, end: int) -> str:
    grid = table.grid()

    header = [_grid_line(grid[r], r, first=0) for r in range(table.n_header_rows)]
    body = [_grid_line(grid[r], r, first=start) for r in range(start, end)]

    return "\n".join([*header, *body])


def _render_table_rows_as_key_value(table: TableInfo, start: int, end: int) -> str:
    if table.n_header_rows == 0:
        return _render_table_rows_as_grid(table, start, end)

    grid = table.grid()
    labels = _column_labels(table, grid)
    lines: list[str] = []
    for r in range(start, end):
        pairs: list[str] = []
        for c, cell in enumerate(grid[r]):
            if r != max(cell.row, start) or c != cell.col:
                continue
            value = cell.text()
            if not value:
                continue
            pairs.append(f"{labels[c]}: {value}" if labels[c] else value)
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _column_labels(table: TableInfo, grid: tuple[tuple[TableCell, ...], ...]) -> list[str]:
    labels: list[str] = []
    for c in range(table.n_cols):
        cells_above: list[TableCell] = []
        for r in range(table.n_header_rows):
            cell = grid[r][c]
            if cell not in cells_above:
                cells_above.append(cell)
        labels.append(" / ".join(cell.text() for cell in cells_above if cell.text()))
    return labels


def _grid_line(row: tuple[TableCell, ...], r: int, first: int) -> str:
    return " | ".join(
        cell.text() if r == max(cell.row, first) and c == cell.col else ""
        for c, cell in enumerate(row)
    )


def _continues_its_marker(previous: Segment | None, segment: TextSegment) -> bool:
    return (
        isinstance(previous, HeadingSegment)
        and previous.title is None
        and previous.marker is not None
        and previous.block_id == segment.block_id
    )
