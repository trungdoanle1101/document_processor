from html import escape
from typing import assert_never

from document_processor.chunking import Chunk
from document_processor.core.models import (
    DocumentPart,
    HeadingSegment,
    ImageSegment,
    TableCell,
    TableSegment,
    TextSegment,
)
from document_processor.pipeline.review import (
    ChunkOrGap,
    Gap,
    ResultSummary,
    ReviewData,
    ReviewNode,
    ReviewWarning,
    build_table_rows,
    node_label,
)

from ._review_page_css import _CSS


def render_review_html(data: ReviewData) -> str:
    entries = "\n".join(
        _render_entry(entry, data) for entry in data.chunks_or_gaps
    )
    
    return _render_page(
        title=data.summary.source.name,
        summary=_render_summary(data.summary),
        outline=_render_outline(data),
        entries=entries
    )


def render_review_frame(data: ReviewData, height_px: int = 640) -> str:
    """The review page inside an iframe, for notebooks: its CSS and ids stay out of the notebook's page"""
    page = render_review_html(data)
    return (
        f'<iframe srcdoc="{escape(page, quote=True)}" '
        f'style="width:100%; height:{height_px}px; border:1px solid #d0d7de; border-radius:6px">'
        "</iframe>"
    )


def _render_entry(entry: ChunkOrGap, data: ReviewData) -> str:
    return _render_chunk(entry, data) if isinstance(entry, Chunk) else _render_gap(entry, data)


def _render_page(title: str, summary: str, outline: str, entries: str) -> str:
    """The whole page: summary on top, outline on the left, chunks and gaps in the main pane"""
    return (
        "<!doctype html>\n"
        '<html lang="vi">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)} · review</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        "</head>\n"
        "<body>\n"
        f"{summary}\n"
        '<div class="layout">\n'
        f"{outline}\n"
        f"<main>\n{entries}\n</main>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )

def _render_gap(gap: Gap, data: ReviewData) -> str:
    """Segments in no chunk: headings a later breadcrumb shows (heading lines), or dropped (a grey stub)"""
    segments = "\n".join(_render_segment(index, None, data) for index in range(gap.start, gap.end))
    if gap.in_breadcrumb:
        return f'<section id="{escape(gap.id)}" class="gap in-breadcrumb {gap.part}">\n{segments}\n</section>'
    count = gap.end - gap.start
    label = f"dropped: {count} segment{'' if count == 1 else 's'} in no chunk and no breadcrumb"
    return (
        f'<section id="{escape(gap.id)}" class="gap dropped {gap.part}">\n'
        f'<div class="gap-label">{label}</div>\n'
        f"{segments}\n"
        "</section>"
    )


def _render_chunk(chunk: Chunk, data: ReviewData) -> str:
    """One chunk as a box: header strip, its segments rendered, and the exact embedded text"""
    over = " over" if chunk.token_count > data.summary.max_tokens else ""
    segments = "\n".join(
        _render_segment(index, chunk.table_rows, data) for index in range(chunk.start, chunk.end)
    )
    return (
        f'<section id="{escape(chunk.id)}" class="chunk {chunk.part}{over}">\n'
        f"{_render_chunk_header(chunk, data.summary.max_tokens)}\n"
        f'<div class="segments">\n{segments}\n</div>\n'
        f"<details><summary>as embedded</summary><pre>{escape(chunk.contextualize())}</pre></details>\n"
        "</section>"
    )


def _render_chunk_header(chunk: Chunk, max_tokens: int) -> str:
    """Id, breadcrumb, table rows of a piece, and a token bar against the budget"""
    breadcrumb = " › ".join(escape(step.label or f"[{step.level}]") for step in chunk.path)
    rows = (
        f' <span class="rows">rows [{chunk.table_rows[0]}, {chunk.table_rows[1]})</span>'
        if chunk.table_rows is not None
        else ""
    )
    width = min(100, round(100 * chunk.token_count / max_tokens))
    return (
        '<div class="chunk-header">'
        f'<span class="chunk-id">{escape(chunk.id)}</span> '
        f'<span class="breadcrumb">{breadcrumb}</span>{rows} '
        f'<span class="tokens"><span class="bar" style="width:{width}%"></span></span> '
        f'<span class="token-count">{chunk.token_count}/{max_tokens}</span>'
        "</div>"
    )


def _render_segment(index: int, table_rows: tuple[int, int] | None, data: ReviewData) -> str:
    """One segment of a chunk or gap; hovering shows its index and block id"""
    segment = data.document.segments[index]
    title = f"segment {index} · {segment.block_id}"
    match segment:
        case HeadingSegment():
            node = data.document.get_node(segment.node_id)
            depth = data.depths[segment.node_id]
            return (
                f'<div class="heading" style="--depth:{depth}" title="{escape(f"{title} · {node.level}")}">'
                f"{escape(node_label(node))}</div>"
            )
        case TextSegment():
            return f'<p title="{escape(title)}">{escape(segment.text)}</p>'
        case TableSegment():
            table = segment.table
            start, end = table_rows if table_rows is not None else (table.n_header_rows, table.n_rows)
            return _render_table(build_table_rows(table, start, end), table.n_header_rows, title)
        case ImageSegment():
            return f'<div class="image" title="{escape(title)}">[image]</div>'
        case _:
            assert_never(segment)


def _render_table(rows: tuple[tuple[TableCell, ...], ...], n_header_rows: int, title: str) -> str:
    """Header rows as <th> in <thead>, the rest as <td>; spans written only when above 1"""
    head = "".join(_render_row(row, "th") for row in rows[:n_header_rows])
    body = "".join(_render_row(row, "td") for row in rows[n_header_rows:])
    thead = f"<thead>{head}</thead>" if head else ""
    return f'<table title="{escape(title)}">{thead}<tbody>{body}</tbody></table>'


def _render_row(cells: tuple[TableCell, ...], tag: str) -> str:
    return "<tr>" + "".join(_render_cell(cell, tag) for cell in cells) + "</tr>"


def _render_cell(cell: TableCell, tag: str) -> str:
    spans = (f' rowspan="{cell.row_span}"' if cell.row_span > 1 else "") + (
        f' colspan="{cell.col_span}"' if cell.col_span > 1 else ""
    )
    paragraphs = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in cell.paragraphs)
    return f"<{tag}{spans}>{paragraphs}</{tag}>"


def _render_summary(summary: ResultSummary) -> str:
    warnings = "\n".join(_render_warning(w) for w in summary.warnings) or "<li>no warnings</li>"
    return (
        '<header class="summary">'
        f"<pre>{escape(summary.format_statistics())}</pre>"
        f'<ul class="warnings">{warnings}</ul>'
        "</header>"
    )


def _render_warning(warning: ReviewWarning) -> str:
    text = f"⚠ {escape(warning.code)}  {escape(warning.message)}"
    if warning.chunk_id is None:
        return f"<li>{text}</li>"
    return f'<li><a href="#{escape(warning.chunk_id)}">{text}</a></li>'


def _render_outline(data: ReviewData) -> str:
    """The outline tree: sections open, items folded; each label links to the first chunk its node overlaps"""
    lines = ['<nav class="outline">']
    lines.append(_render_part_link(data, "front_matter", "front matter"))

    outline = data.outline
    open_details = 0
    for position, node in enumerate(outline):
        while open_details > node.depth:
            lines.append("</details>")
            open_details -= 1
        has_children = position + 1 < len(outline) and outline[position + 1].depth > node.depth
        if has_children:
            opened = " open" if node.is_section else ""
            lines.append(f"<details{opened}><summary>{_render_outline_line(node)}</summary>")
            open_details += 1
        else:
            lines.append(f"<div>{_render_outline_line(node)}</div>")
    lines.extend("</details>" for _ in range(open_details))
    lines.append(_render_part_link(data, "end_matter", "end matter"))
    lines.append("</nav>")
    return "\n".join(line for line in lines if line)


def _render_outline_line(node: ReviewNode) -> str:
    """One node: its label (a link to its first chunk) and how many chunks it spans"""
    kind = "section" if node.is_section else "item"
    count = len(node.overlap_chunk_ids)
    title = escape(f"{node.level} · {count} chunk{'' if count == 1 else 's'}")
    label = escape(node.label)
    if not node.overlap_chunk_ids:
        return f'<span class="{kind}" title="{title}">{label}</span>'
    target = escape(node.overlap_chunk_ids[0])
    spread = f' <span class="count">{count} chunks</span>' if count > 1 else ""
    return f'<a class="{kind}" href="#{target}" title="{title}">{label}</a>{spread}'


def _render_part_link(data: ReviewData, part: DocumentPart, label: str) -> str:
    """A link to the first chunk or gap of front or end matter; empty when the part is empty"""
    for entry in data.chunks_or_gaps:
        if entry.part == part:
            return f'<div class="part"><a href="#{escape(entry.id)}">{label}</a></div>'
    return ""