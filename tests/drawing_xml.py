"""XML builders for <w:drawing> elements, shared by the image tests.

Each drawing declares its own namespaces, so it can be dropped into a <w:r>
of any test document or parsed on its own.
"""

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CHART_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"

DRAWING_NS = (
    f'xmlns:w="{W_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" '
    f'xmlns:pic="{PIC_NS}" xmlns:r="{R_NS}"'
)

# 1 inch = 914400 EMU = 72 pt
EMU_INCH = 914400


def pic(embed: str | None = "rId10", *, link: str | None = None) -> str:
    """One <pic:pic>; `embed` / `link` set a:blip's r:embed / r:link."""
    attrs = ""
    if embed is not None:
        attrs += f' r:embed="{embed}"'
    if link is not None:
        attrs += f' r:link="{link}"'
    return (
        "<pic:pic><pic:nvPicPr><pic:cNvPr id=\"0\" name=\"image.png\"/><pic:cNvPicPr/></pic:nvPicPr>"
        f"<pic:blipFill><a:blip{attrs}/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
        "<pic:spPr/></pic:pic>"
    )


def chart(rid: str = "rId20") -> str:
    """Graphic data of a chart: a drawing that is not a picture."""
    return f'<c:chart xmlns:c="{CHART_NS}" r:id="{rid}"/>'


def drawing(
    *graphics: str,
    anchor: bool = False,
    cx: int | None = EMU_INCH,
    cy: int | None = EMU_INCH // 2,
    extent: bool = True,
    container: bool = True,
) -> str:
    """A <w:drawing>. With no `graphics`, holds one embedded picture (rId10).

    `extent=False` leaves wp:extent out; cx/cy None leave that attribute out.
    `container=False` leaves out wp:inline / wp:anchor (a malformed drawing).
    """
    body = "".join(graphics) if graphics else pic()
    graphic_uri = PIC_NS if "<pic:pic" in body else CHART_NS
    graphic = f'<a:graphic><a:graphicData uri="{graphic_uri}">{body}</a:graphicData></a:graphic>'
    if not container:
        return f"<w:drawing {DRAWING_NS}>{graphic}</w:drawing>"

    ext = ""
    if extent:
        ext_attrs = ""
        if cx is not None:
            ext_attrs += f' cx="{cx}"'
        if cy is not None:
            ext_attrs += f' cy="{cy}"'
        ext = f"<wp:extent{ext_attrs}/>"
    tag = "wp:anchor" if anchor else "wp:inline"
    inner = f'{ext}<wp:docPr id="1" name="Picture 1" descr="image-20250620-095116.png"/>{graphic}'
    return f"<w:drawing {DRAWING_NS}><{tag}>{inner}</{tag}></w:drawing>"