"""translate_filter must select exactly the points Condition.match accepts: checked on real Qdrant."""

from collections.abc import Iterator
from datetime import date

import pytest
from qdrant_client import QdrantClient, models

from document_processor.integrations.qdrant._filters import translate_filter
from document_processor.integrations.qdrant._payload import PointKind, encode_attributes
from document_processor.integrations.qdrant.index import _SUPPORTED_CONDITIONS
from document_processor.retrieval.attributes import (
    AllOf,
    AnyOf,
    AttributeValue,
    BooleanEquals,
    Condition,
    DateBetween,
    ElementCondition,
    EveryElement,
    IntegerIn,
    NoneOf,
    NumberBetween,
    TextIn,
)
from document_processor.retrieval.index import ChunkFilter, UnsupportedConditionError
from tests.retrieval.index_contract import TextStartsWith

# One point per value, all under the attribute "x"; None means the point has no "x"
VALUES: list[AttributeValue | None] = [
    None,
    (),
    "a",
    "b",
    ("a", "b"),
    ("a", "x"),
    3,
    2,
    3.5,
    1.5,
    True,
    False,
    (2, 3),
    (3, 9),
    date(2024, 5, 1),
    date(2024, 7, 1),
]

ELEMENT_CONDITIONS: dict[str, ElementCondition] = {
    "text in": TextIn(values=("a", "b")),
    "integer in": IntegerIn(values=(2, 3)),
    "boolean equals": BooleanEquals(value=True),
    "number at least a fraction": NumberBetween(at_least=2.5),
    "number in a closed range": NumberBetween(at_least=1, at_most=3),
    "number below": NumberBetween(below=3),
    "number strictly between": NumberBetween(above=1.5, below=3.5),
    "date below": DateBetween(below=date(2024, 6, 1)),
    "date in a closed range": DateBetween(
        at_least=date(2024, 5, 1), at_most=date(2024, 6, 30)
    ),
    "date above": DateBetween(above=date(2024, 5, 1)),
}

CONDITIONS: dict[str, Condition] = {
    **ELEMENT_CONDITIONS,
    **{
        f"every element: {name}": EveryElement(condition=c)
        for name, c in ELEMENT_CONDITIONS.items()
    },
    "none of": NoneOf(conditions=(TextIn(values=("a",)),)),
    "any of": AnyOf(conditions=(TextIn(values=("a",)), NumberBetween(above=3))),
    "all of": AllOf(conditions=(IntegerIn(values=(2, 3)), NumberBetween(at_least=3))),
    "nested": AnyOf(
        conditions=(
            NoneOf(conditions=(TextIn(values=("a",)),)),
            EveryElement(condition=IntegerIn(values=(2, 3))),
        )
    ),
}


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    client.create_collection("points", vectors_config={})
    client.upsert(
        "points",
        [
            models.PointStruct(
                id=i,
                vector={},
                payload={
                    "attributes": encode_attributes({} if v is None else {"x": v})
                },
            )
            for i, v in enumerate(VALUES)
        ],
        wait=True,
    )
    yield client
    client.close()


def selected(client: QdrantClient, chunk_filter: ChunkFilter) -> list[int]:
    records, _ = client.scroll(
        "points", scroll_filter=translate_filter(chunk_filter, kind=None), limit=100
    )
    return sorted(int(record.id) for record in records)


@pytest.mark.parametrize("condition", CONDITIONS.values(), ids=CONDITIONS.keys())
def test_qdrant_selects_what_match_accepts(
    client: QdrantClient, condition: Condition
) -> None:
    expected = [i for i, value in enumerate(VALUES) if condition.match(value)]
    assert selected(client, ChunkFilter(attributes={"x": condition})) == expected


def test_a_filter_with_no_parts_selects_every_point(client: QdrantClient) -> None:
    assert selected(client, ChunkFilter()) == list(range(len(VALUES)))


def test_kind_restricts_to_one_kind_of_point() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("points", vectors_config={})
    payloads = [
        {"kind": "chunk", "document_id": "a"},
        {"kind": "entry", "document_id": "a"},
        {"kind": "chunk", "document_id": "b"},
    ]
    client.upsert(
        "points",
        [
            models.PointStruct(id=i, vector={}, payload=p)
            for i, p in enumerate(payloads)
        ],
        wait=True,
    )

    def ids(kind: PointKind | None) -> list[int]:
        f = translate_filter(ChunkFilter(document_ids=("a",)), kind=kind)
        return sorted(
            int(r.id) for r in client.scroll("points", scroll_filter=f, limit=10)[0]
        )

    assert ids("chunk") == [0]
    assert ids("entry") == [1]
    assert ids(None) == [0, 1]


# One example of each supported class: adding a class to _SUPPORTED_CONDITIONS without a
# translation, or without an example here, fails below.
EXAMPLES: dict[type[Condition], Condition] = {
    TextIn: TextIn(values=("a",)),
    IntegerIn: IntegerIn(values=(1,)),
    BooleanEquals: BooleanEquals(value=True),
    NumberBetween: NumberBetween(at_least=1),
    DateBetween: DateBetween(below=date(2024, 1, 1)),
    EveryElement: EveryElement(condition=TextIn(values=("a",))),
    AllOf: AllOf(conditions=(TextIn(values=("a",)),)),
    AnyOf: AnyOf(conditions=(TextIn(values=("a",)),)),
    NoneOf: NoneOf(conditions=(TextIn(values=("a",)),)),
}


def test_every_supported_condition_has_an_example() -> None:
    assert set(EXAMPLES) == set(_SUPPORTED_CONDITIONS)


@pytest.mark.parametrize("condition", EXAMPLES.values(), ids=lambda c: type(c).__name__)
def test_every_supported_condition_translates(condition: Condition) -> None:
    assert isinstance(
        translate_filter(ChunkFilter(attributes={"x": condition}), kind=None),
        models.Filter,
    )


@pytest.mark.parametrize(
    "condition",
    [
        TextStartsWith(prefix="a"),
        EveryElement(condition=TextStartsWith(prefix="a")),
        AnyOf(conditions=(TextStartsWith(prefix="a"),)),
    ],
    ids=["alone", "inside every element", "inside a combinator"],
)
def test_an_unknown_condition_is_refused(condition: Condition) -> None:
    with pytest.raises(UnsupportedConditionError, match="TextStartsWith"):
        translate_filter(ChunkFilter(attributes={"x": condition}), kind=None)
