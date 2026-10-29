"""Tests for integrations/huggingface/_prompts.py: which prompt queries and documents get.

Only prompts the model declares are used, by name, with the model's own
text. An empty prompt counts as no prompt: sentence-transformers 6.2 adds
empty "query" and "document" prompts to every model, and an empty "document"
must not shadow a real "passage" when the convention looks for one. A name
the spec gives may be an empty prompt, which then means no prompt. A model
whose prompt names are all conventional gets them automatically; a model
with any other name must have both roles named in the spec, so a prompt is
never left out silently.

The prompt dictionaries below are what sentence-transformers 6.2 reports as
model.prompts for each kind of model.
"""

import pytest

from document_processor.integrations.huggingface._prompts import (
    CONVENTIONAL_PROMPT_NAMES,
    declared_prompts,
    resolve_prompt_name,
    resolve_prompt_names,
)

BGE_M3 = {"query": "", "document": ""}  # declares none; 6.2 adds the empty pair
E5 = {"query": "query: ", "document": "", "passage": "passage: "}  # "query" and "passage"
QUERY_AND_DOCUMENT = {"query": "query: ", "document": "passage: "}
QUERY_ONLY = {"query": "Represent this sentence for searching relevant passages: ", "document": ""}
CORPUS = {"query": "query: ", "document": "", "corpus": "corpus: "}
NOMIC = {
    "query": "",
    "document": "",
    "search_query": "search_query: ",
    "search_document": "search_document: ",
    "classification": "classification: ",
    "clustering": "clustering: ",
}
WITH_TASK_PROMPTS = {"query": "query: ", "document": "passage: ", "classification": "classify: "}
# codefuse-ai/F2LLM-v2-0.6B declares an instruction for queries and, on purpose, an empty document prompt.
F2LLM = {
    "query": "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: ",
    "document": "",
}


def resolve(prompts: dict[str, str], *, query: str | None = None, document: str | None = None) -> tuple[str | None, str | None]:
    return resolve_prompt_names(prompts, query=query, document=document)


# ============================================================ declared_prompts


def test_empty_prompts_are_dropped() -> None:
    assert declared_prompts(E5) == {"query": "query: ", "passage": "passage: "}


def test_a_model_without_prompts_declares_none() -> None:
    assert declared_prompts(BGE_M3) == {}


def test_prompt_text_is_kept_exactly() -> None:
    # Trailing spaces are part of what the model was trained with.
    assert declared_prompts({"query": "query: "})["query"] == "query: "


def test_a_space_alone_is_a_prompt() -> None:
    # Only the empty string changes nothing; anything else is the model's text.
    assert declared_prompts({"query": " "}) == {"query": " "}


# ============================================================ conventional names


def test_the_convention_is_sentence_transformers_own() -> None:
    assert CONVENTIONAL_PROMPT_NAMES == {"query": ("query",), "document": ("document", "passage", "corpus")}


def test_bge_m3_gets_no_prompts() -> None:
    assert resolve(BGE_M3) == (None, None)


def test_e5_gets_query_and_passage() -> None:
    # sentence-transformers' own encode_document would pick the empty "document"
    # here and encode documents without their prompt.
    assert resolve(E5) == ("query", "passage")


def test_query_and_document() -> None:
    assert resolve(QUERY_AND_DOCUMENT) == ("query", "document")


def test_a_model_that_prompts_only_its_queries() -> None:
    assert resolve(QUERY_ONLY) == ("query", None)


def test_corpus_counts_for_documents() -> None:
    assert resolve(CORPUS) == ("query", "corpus")


def test_document_is_preferred_to_passage_and_corpus() -> None:
    every_name = {"query": "q: ", "document": "d: ", "passage": "p: ", "corpus": "c: "}
    assert resolve(every_name) == ("query", "document")
    assert resolve({"query": "q: ", "passage": "p: ", "corpus": "c: "}) == ("query", "passage")


def test_an_explicit_name_wins_over_the_convention() -> None:
    assert resolve(QUERY_AND_DOCUMENT, document="query") == ("query", "query")


def test_an_explicit_name_must_be_one_the_model_has() -> None:
    with pytest.raises(ValueError, match=r"^query prompt 'question' is not one the model has; it has: document, query$"):
        resolve(QUERY_AND_DOCUMENT, query="question")


def test_naming_an_empty_prompt_means_no_prompt() -> None:
    # bge-m3's "query" is only the empty one 6.2 adds: naming it is allowed and changes nothing.
    assert resolve(BGE_M3, query="query", document="document") == (None, None)


def test_a_model_without_any_prompts_names_none() -> None:
    with pytest.raises(ValueError, match=r"it has: none$"):
        resolve({}, query="query")


# ============================================================ an empty prompt declared on purpose


def test_f2llm_prompts_its_queries_only() -> None:
    assert resolve(F2LLM) == ("query", None)


def test_f2llm_named_explicitly_gives_the_same() -> None:
    assert resolve(F2LLM, query="query", document="document") == ("query", None)


def test_f2llm_documents_with_the_query_instruction_if_asked() -> None:
    # An explicit choice is followed even when it is unusual.
    assert resolve(F2LLM, document="query") == ("query", "query")


# ============================================================ unconventional names


def test_nomic_style_names_must_both_be_named() -> None:
    with pytest.raises(ValueError) as raised:
        resolve(NOMIC)
    assert str(raised.value) == (
        "the model declares prompts with names the convention does not know "
        "(classification, clustering, search_document, search_query; "
        "all: classification, clustering, search_document, search_query); "
        "name the ones to use in the spec: query_prompt_name and document_prompt_name"
    )


def test_naming_one_role_is_not_enough() -> None:
    with pytest.raises(ValueError, match="name the ones to use in the spec: document_prompt_name$"):
        resolve(NOMIC, query="search_query")
    with pytest.raises(ValueError, match="name the ones to use in the spec: query_prompt_name$"):
        resolve(NOMIC, document="search_document")


def test_naming_both_roles_resolves() -> None:
    assert resolve(NOMIC, query="search_query", document="search_document") == ("search_query", "search_document")


def test_a_task_prompt_next_to_conventional_ones_also_requires_names() -> None:
    # The convention would pick query and document, but the model shows it has
    # more than one use; the spec must say which prompts retrieval uses.
    with pytest.raises(ValueError, match=r"\(classification; all: classification, document, query\)"):
        resolve(WITH_TASK_PROMPTS)
    assert resolve(WITH_TASK_PROMPTS, query="query", document="document") == ("query", "document")


def test_names_in_an_unconventional_model_are_still_checked() -> None:
    with pytest.raises(ValueError, match="^document prompt 'search_passage' is not one the model has"):
        resolve(NOMIC, query="search_query", document="search_passage")


# ============================================================ one role at a time


def test_resolving_one_role_on_its_own() -> None:
    assert resolve_prompt_name(E5, "query", None) == "query"
    assert resolve_prompt_name(E5, "document", None) == "passage"
    assert resolve_prompt_name({}, "document", None) is None


def test_none_values_count_as_empty() -> None:
    # sentence-transformers' type for prompts allows None (its placeholder before loading).
    with_none: dict[str, str | None] = {"query": "query: ", "document": None}
    assert declared_prompts(with_none) == {"query": "query: "}
    assert resolve_prompt_names(with_none, query=None, document="document") == ("query", None)