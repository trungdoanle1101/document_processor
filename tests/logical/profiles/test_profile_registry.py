"""Tests for logical/profiles/__init__.py: PROFILES, the registry of built-in profiles.

Importing the package registers every built-in profile under its own name, so a
profile can be chosen by name (from a CLI flag or a config file).
"""

import pytest

from document_processor.logical.profile import Profile
from document_processor.logical.profiles import PROFILES
from document_processor.logical.profiles.vn_regulation import VN_REGULATION


def test_vn_regulation_is_registered() -> None:
    assert PROFILES.get("vn_regulation") is VN_REGULATION


def test_builtin_names() -> None:
    assert PROFILES.names() == ("vn_regulation",)


@pytest.mark.parametrize("name", PROFILES.names())
def test_every_profile_is_registered_under_its_own_name(name: str) -> None:
    profile = PROFILES.get(name)
    assert isinstance(profile, Profile)
    assert profile.name == name


def test_unknown_profile_lists_the_known_ones() -> None:
    with pytest.raises(
        KeyError, match="unknown profile 'vn_regulaton'; known: vn_regulation"
    ):
        PROFILES.get("vn_regulaton")


def test_contains() -> None:
    assert "vn_regulation" in PROFILES
    assert "VN_REGULATION" not in PROFILES


def test_a_profile_can_be_added_without_changing_the_builtins() -> None:
    custom = Profile(name="custom", version="1", levels=None, end_matter=None)
    extended = PROFILES.extend({custom.name: custom})
    assert extended.get("custom") is custom
    assert extended.get("vn_regulation") is VN_REGULATION
    assert "custom" not in PROFILES


def test_a_builtin_name_cannot_be_replaced() -> None:
    other = Profile(name="vn_regulation", version="99", levels=None, end_matter=None)
    with pytest.raises(ValueError, match="profile already defined: vn_regulation"):
        PROFILES.extend({other.name: other})
