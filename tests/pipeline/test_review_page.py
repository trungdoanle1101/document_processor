"""Tests for pipeline/review_page.py: the review page as HTML.

render_review_html(data) returns one self-contained page: the summary (the
statistics in a <pre>, each warning a link to its chunk), the outline tree
(sections open, items with children folded, each label a link to the first
chunk its node overlaps, "N chunks" only when N > 1, front and end matter
links), and one <section> per chunk or gap in document order. A chunk shows
its header strip (id, breadcrumb, table rows, token bar), its segments and the
exact embedded text; a table piece is a <table> with its header rows in
<thead>. A gap in the breadcrumb is heading lines; a dropped gap says so.
All document text is escaped. PipelineResult.write_review_page(path) writes
the page as UTF-8 and returns the path.

render_review_frame(data, height_px=640) is the page for notebooks: one
<iframe> whose srcdoc is the whole page, so its CSS and ids stay out of the
notebook. PipelineResult._repr_html_ returns it.

The page is parsed with html.parser into a small tree, so the tests check
structure, never the exact markup.
"""

from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

from document_processor.chunking import Chunk
from document_processor.core.models import LogicalDocument
from document_processor.pipeline.review import Gap, ReviewData, ReviewWarning, build_review_data
from document_processor.pipeline.review_page import render_review_frame, render_review_html
from tests.chunking.test_outline_chunker import Blocks, regulation
from tests.factories import make_table_block
from tests.pipeline.test_review import MERGED, review

VOID = {"meta", "br", "img", "hr", "link", "input"}


class Element:
    def __init__(self, tag: str, attrs: dict[str, str]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[Element] = []
        self.parts: list[str] = []  # text directly inside, in order

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def text(self) -> str:
        """All text inside, children included, in document order"""
        return "".join(self.parts)

    def find_all(self, tag: str | None = None, cls: str | None = None) -> list["Element"]:
        found: list[Element] = []
        for child in self.children:
            if (tag is None or child.tag == tag) and (cls is None or cls in child.classes):
                found.append(child)
            found.extend(child.find_all(tag, cls))
        return found

    def find(self, tag: str | None = None, cls: str | None = None) -> "Element":
        (only,) = self.find_all(tag, cls)
        return only


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element("#root", {})
        self.stack = [self.root]
        self.doctype: str | None = None

    def handle_decl(self, decl: str) -> None:
        self.doctype = decl

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = Element(tag, {name: value or "" for name, value in attrs})
        self.stack[-1].children.append(element)
        if tag not in VOID:
            self.stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack[-1].tag == tag, f"</{tag}> closes <{self.stack[-1].tag}>"
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        for element in self.stack:
            element.parts.append(data)


def parse(html: str) -> TreeBuilder:
    builder = TreeBuilder()
    builder.feed(html)
    builder.close()
    assert builder.stack == [builder.root], f"unclosed: {[e.tag for e in builder.stack[1:]]}"
    return builder


def page(data: ReviewData) -> Element:
    return parse(render_review_html(data)).root


def entries(root: Element) -> list[Element]:
    return root.find("main").find_all("section")


def with_warnings(data: ReviewData, *warnings: ReviewWarning) -> ReviewData:
    return replace(data, summary=replace(data.summary, warnings=warnings))


def table_document() -> LogicalDocument:
    blocks = Blocks().text("Điều 1. Phạm vi").text("Bảng:")
    blocks.blocks.append(make_table_block(id="pb3", table_info=MERGED))
    return blocks.build()


# ======================================================================
# The page as a whole
# ======================================================================


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_page_is_well_formed(max_tokens: int) -> None:
    builder = parse(render_review_html(review(regulation(), max_tokens)[0]))
    assert builder.doctype is not None and builder.doctype.lower() == "doctype html"


def test_head() -> None:
    root = page(review(regulation())[0])
    html = root.find("html")
    assert html.attrs["lang"] == "vi"
    assert any(meta.attrs.get("charset") == "utf-8" for meta in root.find_all("meta"))
    assert root.find("title").text() == "quy_dinh.docx · review"
    assert root.find("style").text().strip()


def test_layout_holds_summary_outline_and_main() -> None:
    root = page(review(regulation())[0])
    body = root.find("body")
    assert [child.tag for child in body.children] == ["header", "div"]
    assert [child.tag for child in body.find("div", "layout").children] == ["nav", "main"]


# ======================================================================
# Chunks and gaps
# ======================================================================


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_one_section_per_chunk_or_gap_in_order(max_tokens: int) -> None:
    data, _ = review(regulation(), max_tokens)
    assert [section.attrs["id"] for section in entries(page(data))] == [e.id for e in data.chunks_or_gaps]


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_ids_are_unique(max_tokens: int) -> None:
    ids = [element.attrs["id"] for element in _all(page(review(regulation(), max_tokens)[0])) if "id" in element.attrs]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_every_link_points_at_an_element_on_the_page(max_tokens: int) -> None:
    data, chunks = review(regulation(), max_tokens)
    data = with_warnings(data, ReviewWarning(code="chunk_over_budget", message="m", chunk_id=chunks[-1].id))
    root = page(data)
    ids = {element.attrs["id"] for element in _all(root) if "id" in element.attrs}
    hrefs = [a.attrs["href"] for a in root.find_all("a")]
    assert hrefs and all(href.startswith("#") and href[1:] in ids for href in hrefs)


def _all(root: Element) -> list[Element]:
    return root.find_all()


def test_chunk_classes_carry_the_part() -> None:
    data, _ = review(regulation())
    by_id = {section.attrs["id"]: section for section in entries(page(data))}
    for entry in data.chunks_or_gaps:
        assert entry.part in by_id[entry.id].classes


def test_chunk_over_budget_is_marked() -> None:
    data, chunks = review(regulation(), 3)
    by_id = {section.attrs["id"]: section for section in entries(page(data))}
    for chunk in chunks:
        assert ("over" in by_id[chunk.id].classes) == (chunk.token_count > 3)
    assert any(chunk.token_count > 3 for chunk in chunks)


def test_chunk_header_strip() -> None:
    data, chunks = review(regulation())
    chunk = next(c for c in chunks if len(c.path) >= 2)
    section = next(s for s in entries(page(data)) if s.attrs["id"] == chunk.id)
    header = section.find("div", "chunk-header")
    assert header.find("span", "chunk-id").text() == chunk.id
    assert header.find("span", "breadcrumb").text() == " › ".join(step.label for step in chunk.path)
    assert header.find("span", "token-count").text() == f"{chunk.token_count}/200"
    assert header.find_all("span", "rows") == []


def test_token_bar_width_is_capped_at_the_full_bar() -> None:
    data, chunks = review(regulation(), 3)
    over = next(c for c in chunks if c.token_count > 3)
    section = next(s for s in entries(page(data)) if s.attrs["id"] == over.id)
    assert section.find("span", "bar").attrs["style"] == "width:100%"


def test_as_embedded_is_the_exact_contextualized_text() -> None:
    data, chunks = review(regulation(), 8)
    by_id = {section.attrs["id"]: section for section in entries(page(data))}
    for chunk in chunks:
        details = [d for d in by_id[chunk.id].children if d.tag == "details"]
        assert [d.find("summary").text() for d in details] == ["as embedded"]
        assert details[0].find("pre").text() == chunk.contextualize()


def test_segments_carry_their_index_and_block_on_hover() -> None:
    document = regulation()
    data, _ = review(document)
    texts = [p for p in page(data).find("main").find_all("p") if "title" in p.attrs]
    assert texts
    for p in texts:
        index = int(p.attrs["title"].split(" · ")[0].removeprefix("segment "))
        assert p.attrs["title"] == f"segment {index} · {document.segments[index].block_id}"
        assert p.text() == document.segments[index].text  # type: ignore[union-attr]


def test_headings_carry_their_depth_and_level() -> None:
    data, _ = review(regulation(), 3)
    headings = page(data).find("main").find_all("div", "heading")
    assert headings
    for heading in headings:
        index = int(heading.attrs["title"].split(" · ")[0].removeprefix("segment "))
        segment = data.document.segments[index]
        node = data.document.get_node(segment.node_id)  # type: ignore[union-attr]
        assert heading.attrs["style"] == f"--depth:{data.depths[node.id]}"
        assert heading.attrs["title"].endswith(f" · {node.level}")


def test_images_are_placeholders() -> None:
    data, _ = review(regulation())
    images = page(data).find("main").find_all("div", "image")
    assert [image.text() for image in images] == ["[image]"]


# ---------- gaps


def test_gap_in_the_breadcrumb_is_heading_lines() -> None:
    data, _ = review(regulation())
    (gap,) = [e for e in data.chunks_or_gaps if isinstance(e, Gap)]
    section = next(s for s in entries(page(data)) if s.attrs["id"] == gap.id)
    assert {"gap", "in-breadcrumb", "body"} <= section.classes
    assert section.find_all("div", "gap-label") == []
    assert [h.text() for h in section.find_all("div", "heading")] == ["Chương I QUY ĐỊNH CHUNG"]


def test_dropped_gap_says_so_and_shows_what_was_dropped() -> None:
    document = Blocks().image().image().text("Điều 1. Phạm vi").text("Nội dung").build()
    data, _ = review(document)
    section = entries(page(data))[0]
    assert {"gap", "dropped", "front_matter"} <= section.classes
    assert section.find("div", "gap-label").text() == "dropped: 2 segments in no chunk and no breadcrumb"
    assert [image.text() for image in section.find_all("div", "image")] == ["[image]", "[image]"]


def test_dropped_gap_of_one_segment() -> None:
    data, _ = review(Blocks().image().text("Điều 1. Phạm vi").text("Nội dung").build())
    assert entries(page(data))[0].find("div", "gap-label").text().startswith("dropped: 1 segment in")


# ---------- tables


def table_pieces(data: ReviewData) -> list[Element]:
    piece_ids = {e.id for e in data.chunks_or_gaps if isinstance(e, Chunk) and e.table_rows is not None}
    return [s for s in entries(page(data)) if s.attrs["id"] in piece_ids]


def test_every_piece_of_a_split_table_repeats_the_header() -> None:
    data, _ = review(table_document(), 6)
    pieces = table_pieces(data)
    assert len(pieces) > 1
    for piece in pieces:
        table = piece.find("table")
        assert [th.text() for th in table.find("thead").find_all("th")] == ["TT", "Nội dung"]


def test_piece_header_strip_shows_its_rows() -> None:
    data, _ = review(table_document(), 6)
    chunks = [e for e in data.chunks_or_gaps if isinstance(e, Chunk) and e.table_rows is not None]
    for chunk, piece in zip(chunks, table_pieces(data)):
        assert chunk.table_rows is not None
        assert piece.find("span", "rows").text() == f"rows [{chunk.table_rows[0]}, {chunk.table_rows[1]})"


def test_merged_cell_gets_rowspan_cut_to_the_piece() -> None:
    data, _ = review(table_document(), 200)  # the whole table in one chunk
    (table,) = page(data).find_all("table")
    body_rows = table.find("tbody").find_all("tr")
    first_cell = body_rows[0].find_all("td")[0]
    assert (first_cell.text(), first_cell.attrs.get("rowspan")) == ("A", "3")
    # a span of 1 is not written
    assert all(td.attrs.get("rowspan") != "1" and td.attrs.get("colspan") != "1" for td in table.find_all("td"))


def test_table_without_header_rows_has_no_thead() -> None:
    blocks = Blocks().text("Điều 1. Phạm vi").text("Bảng:")
    blocks.blocks.append(make_table_block(id="pb3", table_info=replace(MERGED, n_header_rows=0)))
    data, _ = review(blocks.build(), 200)
    (table,) = page(data).find_all("table")
    assert table.find_all("thead") == []


def test_cell_paragraphs_are_separate() -> None:
    data, _ = review(regulation())
    (table,) = page(data).find_all("table")
    cells = table.find_all("td") + table.find_all("th")
    assert cells and all(all(child.tag == "p" for child in cell.children) for cell in cells)


# ======================================================================
# Summary and warnings
# ======================================================================


def test_summary_shows_the_statistics() -> None:
    data, _ = review(regulation())
    header = page(data).find("header", "summary")
    assert header.find("pre").text() == data.summary.format_statistics()


def test_warning_links_to_its_chunk() -> None:
    data, chunks = review(regulation())
    warning = ReviewWarning(code="chunk_over_budget", message="chunk c2 has 300 tokens", chunk_id=chunks[2].id)
    (item,) = page(with_warnings(data, warning)).find("ul", "warnings").find_all("li")
    link = item.find("a")
    assert link.attrs["href"] == f"#{chunks[2].id}"
    assert link.text() == "⚠ chunk_over_budget  chunk c2 has 300 tokens"


def test_document_wide_warning_is_plain_text() -> None:
    data, _ = review(regulation())
    warning = ReviewWarning(code="no_headings", message="profile found nothing")
    (item,) = page(with_warnings(data, warning)).find("ul", "warnings").find_all("li")
    assert item.find_all("a") == []
    assert item.text() == "⚠ no_headings  profile found nothing"


def test_no_warnings() -> None:
    data, _ = review(regulation())
    (item,) = page(with_warnings(data)).find("ul", "warnings").find_all("li")
    assert item.text() == "no warnings"


# ======================================================================
# Outline
# ======================================================================


def outline_lines(root: Element) -> list[Element]:
    """The <a> or <span> of every node line, in tree order"""
    nav = root.find("nav", "outline")
    return [e for e in nav.find_all() if e.tag in ("a", "span") and ({"section", "item"} & e.classes)]


def test_one_line_per_node_in_order() -> None:
    data, _ = review(regulation())
    assert [line.text() for line in outline_lines(page(data))] == [n.label for n in data.outline]


def test_sections_open_items_folded() -> None:
    # regulation(): Chương I, Điều 1, Điều 2 have children and are sections; khoản 2 has điểm a) and is an item.
    data, _ = review(regulation())
    nav = page(data).find("nav", "outline")
    details = nav.find_all("details")
    states = {d.find_all("summary")[0].text().split(" ")[0]: "open" in d.attrs for d in details}
    assert states == {"Chương": True, "Điều": True, "2.": False}
    assert len(details) == 4  # Chương I, Điều 1, Điều 2, khoản 2


def test_nesting_follows_depth() -> None:
    # A node line sits at the number of <details> around it, not counting the one whose <summary> it is.
    def line_depths(element: Element, depth: int) -> list[int]:
        depths: list[int] = []
        for child in element.children:
            if child.tag == "details":
                depths.append(depth)  # its summary line
                depths.extend(line_depths(child, depth + 1))
            elif child.tag == "div" and "part" not in child.classes:
                depths.append(depth)
        return depths

    data, _ = review(regulation())
    assert line_depths(page(data).find("nav", "outline"), 0) == [n.depth for n in data.outline]


def test_line_links_to_the_first_overlapping_chunk() -> None:
    data, _ = review(regulation(), 3)
    for line, node in zip(outline_lines(page(data)), data.outline):
        assert line.tag == "a" and line.attrs["href"] == f"#{node.overlap_chunk_ids[0]}"
        assert line.attrs["title"].startswith(f"{node.level} · ")


def test_chunk_count_only_when_a_node_spans_several() -> None:
    data, _ = review(regulation(), 3)
    nav = page(data).find("nav", "outline")
    counts = [span.text() for span in nav.find_all("span", "count")]
    assert counts == [f"{len(n.overlap_chunk_ids)} chunks" for n in data.outline if len(n.overlap_chunk_ids) > 1]
    assert counts


def test_node_without_chunks_is_not_a_link() -> None:
    data, _ = review(Blocks().text("Điều 1. Phạm vi").text("Điều 2. Đối tượng").text("Nội dung").build())
    first = outline_lines(page(data))[0]
    assert (first.tag, first.text()) == ("span", "Điều 1. Phạm vi")


def test_front_and_end_matter_links() -> None:
    data, _ = review(regulation())
    parts = page(data).find("nav", "outline").find_all("div", "part")
    first_front = next(e.id for e in data.chunks_or_gaps if e.part == "front_matter")
    first_end = next(e.id for e in data.chunks_or_gaps if e.part == "end_matter")
    assert [(p.text(), p.find("a").attrs["href"]) for p in parts] == [
        ("front matter", f"#{first_front}"),
        ("end matter", f"#{first_end}"),
    ]


def test_no_part_links_without_front_or_end_matter() -> None:
    data, _ = review(Blocks().text("Điều 1. Phạm vi").text("Nội dung").build())
    assert page(data).find("nav", "outline").find_all("div", "part") == []


# ======================================================================
# Escaping
# ======================================================================

HOSTILE = '<script>alert("x")</script> & "quoted"'


def test_document_text_is_escaped_everywhere() -> None:
    document = (
        Blocks()
        .text(f"Điều 1. {HOSTILE}")
        .text(HOSTILE)
        .build()
    )
    data, _ = review(document)
    data = with_warnings(data, ReviewWarning(code="chunk_over_budget", message=HOSTILE, chunk_id="c0"))
    html = render_review_html(data)
    root = parse(html).root
    assert root.find_all("script") == []
    assert HOSTILE not in html
    main_text = root.find("main").text()
    assert main_text.count(HOSTILE) >= 3  # heading, paragraph, as embedded: shown as text
    assert HOSTILE in root.find("nav", "outline").text()
    assert HOSTILE in root.find("ul", "warnings").text()


def test_file_name_is_escaped_in_the_title() -> None:
    data, _ = review(regulation())
    data = replace(data, summary=replace(data.summary, source=Path("a<b>&c.docx")))
    root = page(data)
    assert root.find("title").text() == "a<b>&c.docx · review"


# ======================================================================
# write_review_page
# ======================================================================


def test_write_review_page(tmp_path: Path) -> None:
    from tests.pipeline.test_pipeline import pipeline, write_txt

    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    written = result.write_review_page(tmp_path / "out.html")
    assert written == tmp_path / "out.html"
    assert written.read_text(encoding="utf-8") == render_review_html(build_review_data(result))


def test_write_review_page_accepts_a_string_and_overwrites(tmp_path: Path) -> None:
    from tests.pipeline.test_pipeline import pipeline, write_txt

    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    target = tmp_path / "out.html"
    target.write_text("old", encoding="utf-8")
    assert result.write_review_page(str(target)) == target
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")



# ======================================================================
# render_review_frame and _repr_html_
# ======================================================================


def test_frame_is_one_iframe_and_nothing_else() -> None:
    root = parse(render_review_frame(review(regulation())[0])).root
    assert [child.tag for child in root.children] == ["iframe"]
    assert root.children[0].children == []  # no markup leaks out of the attribute
    assert root.text() == ""


def test_frame_holds_the_whole_page() -> None:
    data, _ = review(regulation())
    (iframe,) = parse(render_review_frame(data)).root.children
    assert iframe.attrs["srcdoc"] == render_review_html(data)  # html.parser unescapes the attribute once


def test_frame_survives_quotes_ampersands_and_tags_in_the_page() -> None:
    document = Blocks().text('Điều 1. "Phạm vi" & <tạm thời>').text('Nội dung "trích dẫn" & ghi chú').build()
    data, _ = review(document)
    frame = render_review_frame(data)
    (iframe,) = parse(frame).root.children
    assert iframe.attrs["srcdoc"] == render_review_html(data)
    assert '"' not in frame.split('srcdoc="', 1)[1].split('" style=', 1)[0]


def test_frame_height() -> None:
    data, _ = review(regulation())
    assert "height:640px" in parse(render_review_frame(data)).root.children[0].attrs["style"]
    assert "height:900px" in parse(render_review_frame(data, height_px=900)).root.children[0].attrs["style"]


def test_repr_html_is_the_frame_of_the_result(tmp_path: Path) -> None:
    from tests.pipeline.test_pipeline import pipeline, write_txt

    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    assert result._repr_html_() == render_review_frame(build_review_data(result))