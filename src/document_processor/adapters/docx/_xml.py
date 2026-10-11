from docx.oxml.ns import nsmap
from lxml import etree
from lxml.etree import _Element


NAMESPACES = {"w": nsmap["w"]}


def _xpath_one_or_none(element: _Element, path: str) -> object | None:
    compiled = etree.XPath(path, namespaces=NAMESPACES, smart_strings=False)
    result = compiled(element)

    if not isinstance(result, list):
        raise TypeError(
            f"XPath: {path}. Unexpected result type from xpath: Expected list, got {type(result).__name__}"
        )

    if len(result) == 0:
        return None

    if len(result) >= 2:
        raise ValueError(f"XPath: {path}. Expected 0 or 1 results, got {len(result)}")

    value: object = result[0]

    return value


def xpath_element_or_none(element: _Element, path: str) -> _Element | None:
    value = _xpath_one_or_none(element, path)

    if value is None:
        return None

    if not isinstance(value, _Element):
        raise TypeError(
            f"XPath: {path}. Expected an _Element, got {value!r} (type: {type(value).__name__})"
        )

    return value


def xpath_str_or_none(element: _Element, path: str) -> str | None:
    value = _xpath_one_or_none(element, path)

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"XPath: {path}. Expected a str, got {value!r} (type: {type(value).__name__})"
        )

    return str(value)


def xpath_int_or_none(element: _Element, path: str) -> int | None:
    value = xpath_str_or_none(element, path)

    if value is None:
        return None

    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"XPath: {path}. Failed to convert {value!r} to int") from e
