import uuid
from collections.abc import Iterable

import pytest

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.filters import Filter, and_, not_, or_
from salemdex_retrieval.in_memory import InMemoryIndex
from salemdex_retrieval.schema import Many, One, Schema


class Chunks(Schema):
    document_id = One(uuid.UUID, indexed=True)
    version = One(int)
    position = One(int)
    groups = Many(uuid.UUID, indexed=True)


class Entries(Schema):
    version = One(int)


class Other(Schema):
    owner_id = One(uuid.UUID)


DOCUMENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_DOCUMENT = uuid.UUID("00000000-0000-0000-0000-000000000002")
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
LEGAL = uuid.UUID("00000000-0000-0000-0000-00000000000b")
FINANCE = uuid.UUID("00000000-0000-0000-0000-00000000000c")


def chunk(
    *,
    document: uuid.UUID = DOCUMENT,
    version: int = 1,
    position: int = 0,
    groups: Iterable[uuid.UUID] = (HR,),
    text: str = "Scope of application",
) -> Chunk:
    """Return a chunk of Chunks whose id is derived from document, version, position."""
    return Chunk(
        id=uuid.uuid5(document, f"{version}:{position}"),
        text=text,
        field_values=[
            Chunks.document_id.value(document),
            Chunks.version.value(version),
            Chunks.position.value(position),
            Chunks.groups.value(groups),
        ],
    )


def unfit_chunk() -> Chunk:
    """Return a chunk lacking every field but the document id."""
    return Chunk(
        id=uuid.uuid4(),
        text="Scope of application",
        field_values=[Chunks.document_id.value(DOCUMENT)],
    )


def index_with(*chunks: Chunk) -> InMemoryIndex:
    index = InMemoryIndex(schema=Chunks)
    index.put(chunks)
    return index


def ids(chunks: Iterable[Chunk]) -> set[uuid.UUID]:
    return {c.id for c in chunks}


EVERYTHING = or_(Chunks.version.eq(1), not_(Chunks.version.eq(1)))


# The schema


def test_an_index_exposes_its_schema() -> None:
    assert InMemoryIndex(schema=Chunks).schema is Chunks


# Storing and finding


def test_find_returns_the_chunks_a_filter_matches() -> None:
    first, second = chunk(position=0), chunk(document=OTHER_DOCUMENT)
    index = index_with(first, second)

    assert ids(index.find(where=Chunks.document_id.eq(DOCUMENT))) == {first.id}


def test_find_returns_the_chunk_as_it_was_stored() -> None:
    stored = chunk(text="Article 1. Scope")
    index = index_with(stored)

    [found] = index.find(where=Chunks.document_id.eq(DOCUMENT))

    assert found == stored
    assert found.text == "Article 1. Scope"


def test_find_returns_nothing_when_no_chunk_matches() -> None:
    index = index_with(chunk())

    assert index.find(where=Chunks.document_id.eq(OTHER_DOCUMENT)) == []


def test_put_accepts_any_iterable_such_as_a_generator() -> None:
    index = InMemoryIndex(schema=Chunks)

    index.put(chunk(position=p) for p in range(3))

    assert len(index.find(where=EVERYTHING)) == 3


def test_putting_the_same_chunks_again_changes_nothing() -> None:
    chunks = [chunk(position=0), chunk(position=1)]
    index = index_with(*chunks)

    index.put(chunks)

    assert ids(index.find(where=EVERYTHING)) == ids(chunks)


def test_put_overwrites_a_stored_chunk_with_the_same_id() -> None:
    index = index_with(chunk(text="old"))

    index.put([chunk(text="new")])

    [found] = index.find(where=EVERYTHING)
    assert found.text == "new"


def test_put_stores_nothing_when_one_chunk_does_not_fit_the_schema() -> None:
    index = InMemoryIndex(schema=Chunks)

    with pytest.raises(SchemaMismatchError):
        index.put([chunk(position=0), chunk(position=1), unfit_chunk()])

    assert index.find(where=EVERYTHING) == []


def test_put_stores_nothing_when_two_chunks_share_an_id() -> None:
    index = InMemoryIndex(schema=Chunks)
    repeated = chunk()

    with pytest.raises(ValueError, match=str(repeated.id)):
        index.put([chunk(position=1), repeated, chunk(text="other text")])

    assert index.find(where=EVERYTHING) == []


# What each filter matches


def test_eq_matches_chunks_whose_value_is_equal() -> None:
    v1, v2 = chunk(version=1), chunk(version=2)
    index = index_with(v1, v2)

    assert ids(index.find(where=Chunks.version.eq(2))) == {v2.id}


def test_any_of_matches_chunks_sharing_at_least_one_value() -> None:
    hr_and_legal = chunk(position=0, groups=[HR, LEGAL])
    finance = chunk(position=1, groups=[FINANCE])
    index = index_with(hr_and_legal, finance)

    assert ids(index.find(where=Chunks.groups.any_of([LEGAL]))) == {hr_and_legal.id}
    assert ids(index.find(where=Chunks.groups.any_of([LEGAL, FINANCE]))) == ids(
        [hr_and_legal, finance]
    )


def test_any_of_with_no_values_matches_no_chunk() -> None:
    index = index_with(chunk(position=0, groups=[HR]), chunk(position=1, groups=[]))

    assert index.find(where=Chunks.groups.any_of([])) == []


def test_any_of_does_not_match_a_chunk_with_no_values() -> None:
    index = index_with(chunk(groups=[]))

    assert index.find(where=Chunks.groups.any_of([HR])) == []


def test_and_matches_chunks_every_filter_matches() -> None:
    wanted = chunk(version=2)
    index = index_with(
        wanted, chunk(version=1), chunk(document=OTHER_DOCUMENT, version=2)
    )

    where = and_(Chunks.document_id.eq(DOCUMENT), Chunks.version.eq(2))

    assert ids(index.find(where=where)) == {wanted.id}


def test_or_matches_chunks_any_filter_matches() -> None:
    hr, legal = chunk(position=0, groups=[HR]), chunk(position=1, groups=[LEGAL])
    index = index_with(hr, legal, chunk(position=2, groups=[FINANCE]))

    where = or_(Chunks.groups.any_of([HR]), Chunks.position.eq(1))

    assert ids(index.find(where=where)) == ids([hr, legal])


def test_not_matches_the_chunks_its_filter_does_not() -> None:
    v1, v2 = chunk(version=1), chunk(version=2)
    index = index_with(v1, v2)

    assert ids(index.find(where=not_(Chunks.version.eq(2)))) == {v1.id}


def test_not_any_of_matches_chunks_with_no_values() -> None:
    no_groups = chunk(position=0, groups=[])
    index = index_with(no_groups, chunk(position=1, groups=[HR]))

    assert ids(index.find(where=not_(Chunks.groups.any_of([HR])))) == {no_groups.id}


# Filters on fields the index doesn't hold


FOREIGN_FILTERS = [
    pytest.param(Entries.version.eq(2), id="same name, another schema"),
    pytest.param(Other.owner_id.eq(DOCUMENT), id="name not in the schema"),
    pytest.param(not_(Entries.version.eq(2)), id="under not_"),
    pytest.param(
        or_(Chunks.version.eq(1), Entries.version.eq(2)),
        id="after a part that already matches",
    ),
]


@pytest.mark.parametrize("where", FOREIGN_FILTERS)
def test_delete_with_a_field_from_another_schema_raises_and_removes_nothing(
    where: Filter,
) -> None:
    index = index_with(chunk(position=0), chunk(position=1))

    with pytest.raises(SchemaMismatchError, match="not in schema Chunks"):
        index.delete(where=where)

    assert len(index.find(where=EVERYTHING)) == 2


@pytest.mark.parametrize("where", FOREIGN_FILTERS)
def test_a_field_from_another_schema_is_rejected_even_in_an_empty_index(
    where: Filter,
) -> None:
    with pytest.raises(SchemaMismatchError):
        InMemoryIndex(schema=Chunks).find(where=where)


def test_the_error_says_when_a_foreign_field_shares_a_name_with_the_schema() -> None:
    with pytest.raises(SchemaMismatchError, match="'version' \\(another schema's"):
        index_with(chunk()).find(where=Entries.version.eq(2))


# Deleting


def test_delete_removes_only_the_chunks_the_filter_matches() -> None:
    kept = chunk(document=OTHER_DOCUMENT)
    index = index_with(chunk(position=0), chunk(position=1), kept)

    index.delete(where=Chunks.document_id.eq(DOCUMENT))

    assert ids(index.find(where=EVERYTHING)) == {kept.id}


def test_delete_after_a_new_version_keeps_only_that_version() -> None:
    v1 = [chunk(version=1, position=p) for p in range(2)]
    v2 = [chunk(version=2, position=p) for p in range(3)]
    other = chunk(document=OTHER_DOCUMENT, version=1)
    index = index_with(*v1, *v2, other)

    index.delete(
        where=and_(Chunks.document_id.eq(DOCUMENT), not_(Chunks.version.eq(2)))
    )

    assert ids(index.find(where=EVERYTHING)) == ids([*v2, other])


def test_delete_matching_nothing_keeps_every_chunk() -> None:
    index = index_with(chunk(position=0), chunk(position=1))

    index.delete(where=Chunks.document_id.eq(OTHER_DOCUMENT))

    assert len(index.find(where=EVERYTHING)) == 2


# Replacing


def test_replace_swaps_the_matched_chunks_for_the_new_ones() -> None:
    old = [chunk(position=p) for p in range(3)]
    other = chunk(document=OTHER_DOCUMENT)
    new = [chunk(version=2, position=p) for p in range(2)]
    index = index_with(*old, other)

    index.replace(where=Chunks.document_id.eq(DOCUMENT), with_=new)

    assert ids(index.find(where=EVERYTHING)) == ids([*new, other])


def test_replace_accepts_new_chunks_that_do_not_match_its_filter() -> None:
    v1 = chunk(version=1)
    v2 = chunk(version=2)
    index = index_with(v1)

    index.replace(
        where=and_(Chunks.document_id.eq(DOCUMENT), Chunks.version.eq(1)), with_=[v2]
    )

    assert ids(index.find(where=EVERYTHING)) == {v2.id}


def test_repeating_a_replace_gives_the_same_chunks() -> None:
    new = [chunk(version=2, position=p) for p in range(2)]
    index = index_with(chunk(version=1))
    where = Chunks.document_id.eq(DOCUMENT)

    index.replace(where=where, with_=new)
    index.replace(where=where, with_=new)

    assert ids(index.find(where=EVERYTHING)) == ids(new)


def test_replace_changes_nothing_when_a_new_chunk_does_not_fit_the_schema() -> None:
    old = [chunk(position=p) for p in range(2)]
    index = index_with(*old)

    with pytest.raises(SchemaMismatchError):
        index.replace(
            where=Chunks.document_id.eq(DOCUMENT),
            with_=[chunk(version=2), unfit_chunk()],
        )

    assert ids(index.find(where=EVERYTHING)) == ids(old)


def test_replace_changes_nothing_when_two_new_chunks_share_an_id() -> None:
    old = chunk(version=1)
    index = index_with(old)

    with pytest.raises(ValueError, match="duplicate"):
        index.replace(
            where=Chunks.document_id.eq(DOCUMENT),
            with_=[chunk(version=2), chunk(version=2, text="other text")],
        )

    assert ids(index.find(where=EVERYTHING)) == {old.id}


def test_replace_changes_nothing_when_its_filter_uses_a_foreign_field() -> None:
    old = chunk(version=1)
    index = index_with(old)

    with pytest.raises(SchemaMismatchError):
        index.replace(where=not_(Entries.version.eq(2)), with_=[chunk(version=2)])

    assert ids(index.find(where=EVERYTHING)) == {old.id}
