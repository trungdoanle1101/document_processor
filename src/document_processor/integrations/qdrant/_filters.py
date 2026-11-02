from datetime import UTC, date, datetime

from qdrant_client import models

from document_processor.integrations.qdrant._payload import (
    CHUNK_KEY,
    DOCUMENT_ID,
    KIND,
    NUMBER_KINDS,
    REFERENCE_KEYS,
    PointKind,
    attribute_key,
    encode_chunk_key,
)
from document_processor.retrieval.attributes import (
    AllOf,
    AnyOf,
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


def translate_filter(
    chunk_filter: ChunkFilter, *, kind: PointKind | None
) -> models.Filter:
    """chunk_filter as a Qdrant filter, on points of one kind, or both if kind is None."""
    filters: list[models.Condition] = [_to_qdrant_filter(chunk_filter)]
    if kind is not None:
        filters.append(
            models.FieldCondition(key=KIND, match=models.MatchValue(value=kind))
        )
    return models.Filter(must=filters)


def _to_qdrant_filter(chunk_filter: ChunkFilter) -> models.Filter:
    """chunk_filter as a Qdrant filter"""
    document_id_cond = (
        models.FieldCondition(
            key=DOCUMENT_ID,
            match=models.MatchAny(any=list(chunk_filter.document_ids)),
        )
        if chunk_filter.document_ids is not None
        else None
    )
    chunk_key_cond = (
        models.FieldCondition(
            key=CHUNK_KEY,
            match=models.MatchAny(
                any=[encode_chunk_key(key) for key in chunk_filter.chunk_keys]
            ),
        )
        if chunk_filter.chunk_keys is not None
        else None
    )
    reference_key_cond = (
        models.FieldCondition(
            key=REFERENCE_KEYS,
            match=models.MatchAny(any=list(chunk_filter.reference_keys)),
        )
        if chunk_filter.reference_keys is not None
        else None
    )

    qdrant_conditions: list[models.Condition] = [
        cond
        for cond in (document_id_cond, chunk_key_cond, reference_key_cond)
        if cond is not None
    ]

    for name, cond in chunk_filter.attributes.items():
        qdrant_conditions.append(_to_qdrant_condition(cond, name))

    return models.Filter(must=qdrant_conditions)


def _to_qdrant_condition(condition: Condition, name: str) -> models.Condition:
    match condition:
        case ElementCondition():
            return _positive(condition, name)

        case AnyOf(conditions=conditions):
            return models.Filter(
                should=[_to_qdrant_condition(c, name) for c in conditions]
            )
        case AllOf(conditions=conditions):
            return models.Filter(
                must=[_to_qdrant_condition(c, name) for c in conditions]
            )
        case NoneOf(conditions=conditions):
            return models.Filter(
                must_not=[_to_qdrant_condition(c, name) for c in conditions]
            )
        case EveryElement(condition=inner):
            return models.Filter(
                must=[_positive(inner, name)], must_not=[_negative(inner, name)]
            )
        case _:
            raise UnsupportedConditionError(
                f"no Qdrant translation for {type(condition).__name__}"
            )


def _positive(condition: ElementCondition, name: str) -> models.Condition:
    match condition:
        case TextIn(values=values):
            return models.FieldCondition(
                key=attribute_key("text", name), match=models.MatchAny(any=list(values))
            )
        case IntegerIn(values=values):
            return models.FieldCondition(
                key=attribute_key("integer", name),
                match=models.MatchAny(any=list(values)),
            )
        case BooleanEquals(value=value):
            return models.FieldCondition(
                key=attribute_key("boolean", name), match=models.MatchValue(value=value)
            )
        case NumberBetween(
            at_least=at_least, at_most=at_most, above=above, below=below
        ):
            r = models.Range(lt=below, gt=above, gte=at_least, lte=at_most)
            return models.Filter(
                should=[
                    models.FieldCondition(key=attribute_key(kind, name), range=r)
                    for kind in NUMBER_KINDS
                ]
            )
        case DateBetween(at_least=at_least, at_most=at_most, above=above, below=below):
            return models.FieldCondition(
                key=attribute_key("date", name),
                range=models.DatetimeRange(
                    lt=_encode_date_bound(below),
                    gt=_encode_date_bound(above),
                    gte=_encode_date_bound(at_least),
                    lte=_encode_date_bound(at_most),
                ),
            )
        case _:
            raise UnsupportedConditionError(
                f"no Qdrant translation for {type(condition).__name__}"
            )


def _negative(condition: ElementCondition, name: str) -> models.Condition:
    match condition:
        case TextIn(values=values):
            return models.FieldCondition(
                key=attribute_key("text", name),
                match=models.MatchExcept(**{"except": list(values)}),
            )
        case IntegerIn(values=values):
            return models.FieldCondition(
                key=attribute_key("integer", name),
                match=models.MatchExcept(**{"except": list(values)}),
            )
        case BooleanEquals(value=value):
            return models.FieldCondition(
                key=attribute_key("boolean", name),
                match=models.MatchValue(value=not value),
            )
        case NumberBetween(
            at_least=at_least, at_most=at_most, above=above, below=below
        ):
            outside_numbers: list[models.Range] = []
            if at_least is not None:
                outside_numbers.append(models.Range(lt=at_least))
            if above is not None:
                outside_numbers.append(models.Range(lte=above))
            if at_most is not None:
                outside_numbers.append(models.Range(gt=at_most))
            if below is not None:
                outside_numbers.append(models.Range(gte=below))
            return models.Filter(
                should=[
                    models.FieldCondition(key=attribute_key(kind, name), range=r)
                    for kind in NUMBER_KINDS
                    for r in outside_numbers
                ]
            )
        case DateBetween(at_least=at_least, at_most=at_most, above=above, below=below):
            outside_dates: list[models.DatetimeRange] = []
            if at_least is not None:
                outside_dates.append(
                    models.DatetimeRange(lt=_encode_date_bound(at_least))
                )
            if above is not None:
                outside_dates.append(
                    models.DatetimeRange(lte=_encode_date_bound(above))
                )
            if at_most is not None:
                outside_dates.append(
                    models.DatetimeRange(gt=_encode_date_bound(at_most))
                )
            if below is not None:
                outside_dates.append(
                    models.DatetimeRange(gte=_encode_date_bound(below))
                )
            return models.Filter(
                should=[
                    models.FieldCondition(key=attribute_key("date", name), range=r)
                    for r in outside_dates
                ]
            )
        case _:
            raise UnsupportedConditionError(
                f"no Qdrant translation for {type(condition).__name__}"
            )


def _encode_date_bound(bound: date | None) -> datetime | None:
    return (
        None
        if bound is None
        else datetime(bound.year, bound.month, bound.day, tzinfo=UTC)
    )
