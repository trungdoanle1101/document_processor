_PUA_BASE = 0xF000

_SYMBOL_FONTS: dict[str, dict[int, str]] = {
    "symbol":    {0xB7: "•"},
    "wingdings": {0xA7: "▪", 0x76: "❖", 0xD8: "➢", 0xFC: "✓",
                  0xFB: "✗", 0xFE: "☑", 0xFD: "☒", 0x6C: "●", 0x6E: "■", 0x71: "❑"},
    "courier new": {0x6F: "◦"}
}


def map_symbol_text(text: str, font: str | None) -> str:
    """Replace bullet font characters with their Unicode equivalents; unknown ones stay."""
    table = _SYMBOL_FONTS.get(font.casefold()) if font else None
    if table is None:
        return text
    out = []
    for ch in text:
        code = ord(ch)
        if _PUA_BASE <= code <= _PUA_BASE + 0xFF:   # Word's private-use form, U+F0xx
            code -= _PUA_BASE
        out.append(table.get(code, ch))
    return "".join(out)