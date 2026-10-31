"""Tests for logical/registry.py: a read-only name -> item lookup.

A registry never changes after it is built. ``extend`` returns a new registry
and refuses to replace a name that already exists, so a profile can add its own
items without touching (or silently overriding) the built-ins. ``replace`` is
the opposite: it returns a new registry with existing entries swapped, and
refuses a name that is not registered, so a typo fails instead of adding an
entry nobody looks up.
"""

from types import MappingProxyType

import pytest

from document_processor.core.registry import Registry


def fruit_registry() -> Registry[int]:
    return Registry("fruit", {"banana": 2, "apple": 1, "cherry": 3})


# ---------- lookup


def test_get_returns_the_item() -> None:
    registry = fruit_registry()
    assert registry.get("apple") == 1
    assert registry.get("cherry") == 3


def test_get_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        fruit_registry().get("durian")


def test_unknown_name_error_names_the_kind_the_name_and_the_known_names_sorted() -> (
    None
):
    with pytest.raises(KeyError) as excinfo:
        fruit_registry().get("durian")
    message = str(excinfo.value)
    assert "fruit" in message
    assert "'durian'" in message
    assert "apple, banana, cherry" in message


def test_get_is_case_sensitive() -> None:
    with pytest.raises(KeyError):
        fruit_registry().get("Apple")


def test_names_are_sorted() -> None:
    assert fruit_registry().names() == ("apple", "banana", "cherry")


def test_names_is_a_tuple() -> None:
    assert isinstance(fruit_registry().names(), tuple)


def test_contains_checks_names() -> None:
    registry = fruit_registry()
    assert "apple" in registry
    assert "durian" not in registry
    assert 1 not in registry  # type: ignore[operator]  # items are not names


def test_empty_registry() -> None:
    registry: Registry[int] = Registry("fruit", {})
    assert registry.names() == ()
    assert "apple" not in registry
    with pytest.raises(KeyError):
        registry.get("apple")


def test_accepts_any_mapping() -> None:
    registry = Registry("fruit", MappingProxyType({"apple": 1}))
    assert registry.get("apple") == 1


# ---------- read-only


def test_changing_the_source_dict_does_not_change_the_registry() -> None:
    source = {"apple": 1}
    registry = Registry("fruit", source)
    source["banana"] = 2
    source["apple"] = 100
    assert registry.names() == ("apple",)
    assert registry.get("apple") == 1


def test_has_no_method_that_adds_items_in_place() -> None:
    registry = fruit_registry()
    for name in ("register", "add", "set", "update", "__setitem__", "__delitem__"):
        assert not hasattr(registry, name), name


# ---------- extend


def test_extend_returns_a_registry_with_both_sets_of_items() -> None:
    extended = fruit_registry().extend({"durian": 4})
    assert extended.names() == ("apple", "banana", "cherry", "durian")
    assert extended.get("durian") == 4
    assert extended.get("apple") == 1


def test_extend_returns_a_new_registry_and_leaves_the_original_unchanged() -> None:
    original = fruit_registry()
    extended = original.extend({"durian": 4})
    assert extended is not original
    assert "durian" not in original
    assert original.names() == ("apple", "banana", "cherry")


def test_extend_with_nothing() -> None:
    assert fruit_registry().extend({}).names() == ("apple", "banana", "cherry")


def test_extend_twice() -> None:
    extended = fruit_registry().extend({"durian": 4}).extend({"elderberry": 5})
    assert extended.names() == ("apple", "banana", "cherry", "durian", "elderberry")


def test_extend_keeps_the_kind_in_error_messages() -> None:
    extended = fruit_registry().extend({"durian": 4})
    with pytest.raises(KeyError, match="fruit"):
        extended.get("fig")


def test_extend_refuses_to_replace_an_existing_name() -> None:
    with pytest.raises(ValueError, match="fruit already defined: apple"):
        fruit_registry().extend({"apple": 100})


def test_extend_lists_every_clash_sorted() -> None:
    with pytest.raises(ValueError, match="already defined: apple, cherry$"):
        fruit_registry().extend({"cherry": 30, "durian": 4, "apple": 10})


def test_failed_extend_changes_nothing() -> None:
    original = fruit_registry()
    with pytest.raises(ValueError):
        original.extend({"apple": 100, "durian": 4})
    assert original.get("apple") == 1
    assert "durian" not in original


def test_changing_the_extension_dict_does_not_change_the_new_registry() -> None:
    extra = {"durian": 4}
    extended = fruit_registry().extend(extra)
    extra["durian"] = 400
    extra["fig"] = 6
    assert extended.get("durian") == 4
    assert "fig" not in extended


# ---------- replace


def test_replace_swaps_the_item() -> None:
    assert fruit_registry().replace({"apple": 10}).get("apple") == 10


def test_replace_keeps_the_other_items() -> None:
    replaced = fruit_registry().replace({"apple": 10})
    assert (replaced.get("banana"), replaced.get("cherry")) == (2, 3)
    assert replaced.names() == ("apple", "banana", "cherry")


def test_replace_several_at_once() -> None:
    replaced = fruit_registry().replace({"apple": 10, "cherry": 30})
    assert [replaced.get(name) for name in replaced.names()] == [10, 2, 30]


def test_replace_leaves_the_original_unchanged() -> None:
    original = fruit_registry()
    original.replace({"apple": 10})
    assert original.get("apple") == 1


def test_replace_with_nothing_is_the_same_lookup() -> None:
    replaced = fruit_registry().replace({})
    assert [replaced.get(name) for name in replaced.names()] == [1, 2, 3]


def test_replace_refuses_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown fruit: durian"):
        fruit_registry().replace({"durian": 4})


def test_replace_lists_every_unknown_name_sorted() -> None:
    with pytest.raises(KeyError, match="unknown fruit: durian, fig"):
        fruit_registry().replace({"fig": 6, "apple": 10, "durian": 4})


def test_failed_replace_changes_nothing() -> None:
    original = fruit_registry()
    with pytest.raises(KeyError):
        original.replace({"apple": 10, "durian": 4})
    assert original.get("apple") == 1
    assert "durian" not in original


def test_replaced_registry_keeps_its_kind() -> None:
    with pytest.raises(KeyError, match="unknown fruit 'fig'"):
        fruit_registry().replace({"apple": 10}).get("fig")


def test_changing_the_replacement_dict_does_not_change_the_new_registry() -> None:
    replacement = {"apple": 10}
    replaced = fruit_registry().replace(replacement)
    replacement["apple"] = 100
    assert replaced.get("apple") == 10
