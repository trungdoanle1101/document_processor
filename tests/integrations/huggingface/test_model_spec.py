"""Tests for integrations/huggingface/model_specs.py: HuggingFaceModelSpec, BGE_M3 and the registry.

A spec holds choices, never facts about the model. It names the exact model
(a ModelReference) and, only for models whose prompt names aren't
conventional, which of the model's declared prompts to use for queries and
for passages. Prompts are chosen by name ("query"), never by text ("query: "):
the text is a fact of the model, read from its folder, so it cannot be
mistyped, and a name can be checked against what the model declares. That
check needs the folder, so it happens at load, not here. Pooling,
normalization and the maximum length are not in the spec at all:
sentence-transformers reads them from the model.

So a preset like BGE_M3 is just a reference.

The package's __init__ exports only reference.py and model_specs.py, so
listing the presets never imports torch, transformers or huggingface_hub.
"""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest

from document_processor.core.registry import Registry
from document_processor.integrations import huggingface
from document_processor.integrations.huggingface import (
    BGE_M3,
    HUGGINGFACE_MODEL_SPECS,
    HuggingFaceModelSpec,
    ModelReference,
)

COMMIT = "5617a9f61b028005a4858fdac845db406aefb181"
E5 = ModelReference(hub_model_id="intfloat/multilingual-e5-large", commit_hash="0" * 40)


def make_spec(**overrides: object) -> HuggingFaceModelSpec:
    values: dict[str, object] = {"reference": E5}
    values.update(overrides)
    return HuggingFaceModelSpec(**values)  # type: ignore[arg-type]


# ============================================================ the spec


def test_a_reference_alone_is_a_spec() -> None:
    spec = HuggingFaceModelSpec(reference=E5)
    assert spec.reference == E5
    assert (spec.query_prompt_name, spec.passage_prompt_name) == (None, None)


def test_every_choice_can_be_made() -> None:
    spec = HuggingFaceModelSpec(reference=E5, query_prompt_name="query", passage_prompt_name="document")
    assert (spec.query_prompt_name, spec.passage_prompt_name) == ("query", "document")


def test_one_prompt_without_the_other() -> None:
    # Some models prompt only their queries.
    spec = make_spec(query_prompt_name="query")
    assert (spec.query_prompt_name, spec.passage_prompt_name) == ("query", None)


def test_the_spec_holds_choices_only() -> None:
    # Facts (pooling, normalization, maximum length, prompt text) come from the
    # model; a field for one here could contradict the model.
    assert [field.name for field in fields(HuggingFaceModelSpec)] == [
        "reference",
        "query_prompt_name",
        "passage_prompt_name",
    ]


def test_the_reference_is_required() -> None:
    with pytest.raises(TypeError, match="reference"):
        HuggingFaceModelSpec()  # type: ignore[call-arg]


def test_spec_takes_keywords_only() -> None:
    with pytest.raises(TypeError):
        HuggingFaceModelSpec(E5, "query")  # type: ignore[call-arg]


def test_spec_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        BGE_M3.query_prompt_name = "query"  # type: ignore[misc]


def test_specs_compare_by_value() -> None:
    assert make_spec() == make_spec()
    assert make_spec() != make_spec(query_prompt_name="query")
    assert make_spec() != make_spec(passage_prompt_name="passage")
    assert make_spec() != make_spec(reference=ModelReference(hub_model_id=E5.hub_model_id, commit_hash="1" * 40))


def test_specs_can_be_dictionary_keys() -> None:
    # Frozen and all fields hashable, so a loaded embedder can be cached by its spec.
    spec = make_spec(query_prompt_name="query")
    assert {spec: "loaded"}[make_spec(query_prompt_name="query")] == "loaded"


# ============================================================ refusals


@pytest.mark.parametrize("field_name", ["query_prompt_name", "passage_prompt_name"])
def test_an_empty_prompt_name_is_refused(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"^{field_name} must not be empty; use None to use no prompt$"):
        make_spec(**{field_name: ""})


def test_prompt_names_are_checked_against_the_model_at_load_not_here() -> None:
    # Which prompts exist is a fact of the downloaded folder.
    assert make_spec(query_prompt_name="no-such-prompt").query_prompt_name == "no-such-prompt"


def test_a_bad_reference_is_refused_by_the_reference() -> None:
    with pytest.raises(ValueError, match="commit_hash"):
        make_spec(reference=ModelReference(hub_model_id="BAAI/bge-m3", commit_hash="main"))


# ============================================================ BGE_M3


def test_bge_m3_is_pinned_to_one_commit() -> None:
    # Changing the commit means a new folder and re-indexing: a deliberate edit,
    # so this test changes with it.
    assert BGE_M3.reference == ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT)


def test_bge_m3_names_no_prompts() -> None:
    # It declares none; with conventional names it would not need to either.
    assert (BGE_M3.query_prompt_name, BGE_M3.passage_prompt_name) == (None, None)


# ============================================================ the registry


def test_registry_lists_bge_m3() -> None:
    assert HUGGINGFACE_MODEL_SPECS.names() == ("bge-m3",)
    assert HUGGINGFACE_MODEL_SPECS.get("bge-m3") is BGE_M3


def test_unknown_spec_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match=r"unknown embedding model spec 'e5'; known: bge-m3"):
        HUGGINGFACE_MODEL_SPECS.get("e5")


def test_registry_can_be_extended_with_another_spec() -> None:
    e5 = make_spec(query_prompt_name="query", passage_prompt_name="passage")
    specs: Registry[HuggingFaceModelSpec] = HUGGINGFACE_MODEL_SPECS.extend({"e5": e5})
    assert specs.names() == ("bge-m3", "e5")
    assert specs.get("e5") is e5
    assert HUGGINGFACE_MODEL_SPECS.names() == ("bge-m3",)


def test_two_names_may_share_a_model_with_different_choices() -> None:
    # One set of weights, two uses: here, documents encoded with the query prompt.
    symmetric = HuggingFaceModelSpec(reference=BGE_M3.reference, query_prompt_name="query", passage_prompt_name="query")
    specs = HUGGINGFACE_MODEL_SPECS.extend({"bge-m3-symmetric": symmetric})
    assert specs.get("bge-m3-symmetric").reference == specs.get("bge-m3").reference
    assert specs.get("bge-m3-symmetric") != specs.get("bge-m3")


# ============================================================ the package


def test_package_exports_the_reference_and_the_specs() -> None:
    assert sorted(huggingface.__all__) == [
        "BGE_M3",
        "HUGGINGFACE_MODEL_SPECS",
        "HuggingFaceModelSpec",
        "MODEL_REFERENCE_FILE",
        "ModelReference",
        "read_model_reference",
        "write_model_reference",
    ]


def test_importing_the_package_loads_no_heavy_library() -> None:
    # A fresh interpreter: this test process may already have imported them.
    code = (
        "import sys, document_processor.integrations.huggingface; "
        "print(sorted(name for name in ('torch', 'transformers', 'huggingface_hub', 'sentence_transformers')"
        " if name in sys.modules))"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "[]"