"""The pieces of a source that an index stores and search returns."""

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, assert_never

from salemdex_retrieval.schema import Field, FieldValue, ManyValues, OneValue, Schema


class SchemaMismatchError(ValueError):
    """Raised when a chunk, a filter or a stored index doesn't fit its schema.

    A chunk's field values don't match the schema's fields, a filter uses a field
    that isn't one of them, or what a backend holds contradicts the schema (a stored
    point, a payload index of the wrong type). The schema itself is fine (that would be
    a `SchemaError`), so retrying the same call can't help.
    """


@dataclass(frozen=True, kw_only=True, init=False)
class Chunk:
    """A piece of a source, as an index stores it and search returns it.

    Attributes:
        id: Unique across the whole index. Derive it from what identifies the piece
            (e.g. document, version, position), so writing it again overwrites it.
        text: The source's own words, returned unchanged by search. Left out of
            `repr`, so it can't reach logs.
        embedded_text: What the model embeds: the text with context added, or the
            text itself. Left out of `repr`.
        field_values: The chunk's value for each field, keyed by the field: one value
            for a `One`, a tuple for a `Many`. Read-only.
    """

    id: uuid.UUID
    text: str = field(repr=False)
    embedded_text: str = field(repr=False)
    field_values: Mapping[Field[Any], Any]

    def __init__(
        self,
        *,
        id: uuid.UUID,
        text: str,
        embedded_text: str | None = None,
        field_values: Iterable[FieldValue],
    ) -> None:
        """Build a chunk, keeping its own read-only copy of the field values.

        Args:
            id: Unique across the whole index.
            text: The source's own words.
            embedded_text: What the model embeds, such as the headings path followed
                by the text. Left out, the text itself is embedded.
            field_values: One value per field, built by `One.value` and `Many.value`.

        Raises:
            TypeError: `id` isn't a `uuid.UUID`.
            ValueError: Two field values have one name: the same field twice, or
                fields from different schemas.
        """
        if type(id) is not uuid.UUID:
            raise TypeError("id must be of type uuid.UUID")

        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self, "embedded_text", embedded_text if embedded_text is not None else text
        )
        by_field: dict[Field[Any], Any] = {}
        seen_names: set[str] = set()
        for field_value in field_values:
            if field_value.field.name in seen_names:
                raise ValueError(f"duplicate Field {field_value.field.name}")
            seen_names.add(field_value.field.name)

            match field_value:
                case OneValue(field=key, value=value):
                    by_field[key] = value
                case ManyValues(field=key, values=values):
                    by_field[key] = values
                case _:
                    assert_never(field_value)

        object.__setattr__(self, "field_values", MappingProxyType(by_field))

    def check_fits(self, schema: type[Schema]) -> None:
        """Check that the chunk has a value for each of `schema`'s fields, and no other.

        Fields are compared as objects, not by name: a field from another schema with
        the same name doesn't fit. An empty `Many` counts as a value.

        Args:
            schema: The schema of the index the chunk is written to or read from.

        Raises:
            SchemaMismatchError: The chunk holds fields `schema` doesn't have, or lacks
                some it has. One error names the chunk id and every such field.
        """
        schema_fields = schema.fields()
        extra = [f for f in self.field_values if schema_fields.get(f.name) is not f]
        missing = [f for f in schema_fields.values() if f not in self.field_values]
        if not extra and not missing:
            return

        problems = []
        if extra:
            problems.append(
                f"not in {schema.__name__}: {_describe_extra(extra, schema)}"
            )
        if missing:
            problems.append(f"missing: {', '.join(f.name for f in missing)}")
        raise SchemaMismatchError(f"chunk {self.id}: {'; '.join(problems)}")


def _describe_extra(extra: list[Field[Any]], schema: type[Schema]) -> str:
    """Return the extra fields' names, marking any that share a name with `schema`'s."""
    return ", ".join(
        f"{f.name} (another schema's field)" if f.name in schema.fields() else f.name
        for f in extra
    )
