from lxml.etree import _Element
from lxml import etree
from docx.oxml.ns import nsmap

ALIGNMENT_MAPPING = {
    "left": "left",  # Word convention
    "start": "left",  # Libreoffice convention
    "right": "right",  # Word convention
    "end": "right",  # Libreoffice convention
    "center": "center",
    "both": "justified",
}

NAMESPACES = {
    "w": nsmap["w"]
}


JC_VAL_XPATH = "./w:pPr/w:jc/@w:val"


def xpath_str_or_none(element: _Element, path: str) -> str | None:
    e = etree.XPath(path, namespaces=NAMESPACES, smart_strings=False)
    result = e(element)

    if not isinstance(result, list):
        raise TypeError(f"XPath: {path}. Unexpected result type from xpath: Expected list, got {type(result).__name__}")

    if len(result) == 0:
        return None

    if len(result) >= 2:
        raise ValueError(f"XPath: {path}. Expected 0 or 1 results, got {len(result)}")
    
    value = result[0]
    
    if not isinstance(value, str):
        raise TypeError(f"XPath: {path}. Expected a str, got {value!r} (type: {type(value).__name__})")

    return str(value)


def xpath_int_or_none(element: _Element, path: str) -> int | None:
    value = xpath_str_or_none(element, path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"XPath: {path}. Failed to convert {value!r} to int") from e 


def read_jc(element: _Element) -> str | None:
    return xpath_str_or_none(element=element, path=JC_VAL_XPATH)
