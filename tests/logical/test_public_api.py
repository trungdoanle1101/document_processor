"""The logical package's public API: every name in __all__ is importable from the package."""

import document_processor.logical as logical
from document_processor.logical.build import build_logical_document
from document_processor.logical.profile import Profile


def test_every_public_name_is_importable() -> None:
    missing = [name for name in logical.__all__ if not hasattr(logical, name)]
    assert missing == []


def test_names_are_the_same_objects_as_in_their_modules() -> None:
    assert logical.build_logical_document is build_logical_document
    assert logical.Profile is Profile


def test_built_in_profile_is_registered() -> None:
    assert logical.PROFILES.get("vn_regulation") is logical.VN_REGULATION


def test_internals_are_not_exported() -> None:
    for internal in (
        "find_candidates",
        "narrow_candidates",
        "choose_headings",
        "Candidate",
        "BlockCandidates",
    ):
        assert internal not in logical.__all__, internal
