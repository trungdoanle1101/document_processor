from document_processor.core.models import LogicalDocument, heading_label


def format_outline(
    document: LogicalDocument, sections_only: bool = False, indent: int = 2
) -> str:
    outline_builder: list[str] = []

    if document.body_start == 0:
        outline_builder.append("front matter  none")
    else:
        outline_builder.append(f"front matter  [0, {document.body_start})")

    depth: dict[str | None, int] = {None: -1}

    for node in document.nodes:
        depth[node.id] = depth[node.parent_id] + 1
        if sections_only and not node.is_section:
            continue
        label = heading_label(node.marker, node.title) or f"[{node.level}]"
        line = f"{indent * depth[node.id] * ' '}{label}  [{node.start}, {node.end})"
        outline_builder.append(line)

    if document.body_end == len(document.segments):
        outline_builder.append("end matter  none")
    else:
        outline_builder.append(
            f"end matter  [{document.body_end}, {len(document.segments)})"
        )

    return "\n".join(outline_builder)
