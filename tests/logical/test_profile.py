"""Tests for logical/profile.py: LevelSpec and Profile validation."""

import dataclasses
import re

import pytest

from document_processor.logical.profile import LevelSpec, Profile

CHUONG = r"Chương\s+(?P<number>[IVXLC]+)\b"
MUC = r"Mục\s+(?P<number>[IVXLC]+|\d+)\b"
DIEU = r"Điều\s+(?P<number>\d+)[.:]"
KHOAN = r"(?P<number>\d+)\.(?!\d)"
DIEM = r"(?P<number>[a-zđ])\)"


def level(
    name: str,
    rank: int,
    pattern: str = DIEU,
    *,
    title_on_next_block: bool = False,
    required_ancestor: str | None = None,
) -> LevelSpec:
    return LevelSpec(
        name=name,
        rank=rank,
        pattern=pattern,
        title_on_next_block=title_on_next_block,
        required_ancestor=required_ancestor,
    )


def vn_levels() -> tuple[LevelSpec, ...]:
    return (
        level("chuong", 0, CHUONG, title_on_next_block=True),
        level("muc", 1, MUC, title_on_next_block=True),
        level("dieu", 2, DIEU),
        level("khoan", 3, KHOAN, required_ancestor="dieu"),
        level("diem", 4, DIEM, required_ancestor="dieu"),
    )


def profile(
    levels: tuple[LevelSpec, ...] | None = None,
    *,
    use_default_levels: bool = True,
    name: str = "vn_regulation",
    version: str = "1.0.0",
    end_matter_pattern: str | None = r"Nơi nhận",
) -> Profile:
    if levels is None and use_default_levels:
        levels = vn_levels()
    return Profile(
        name=name,
        version=version,
        levels=levels,
        end_matter_pattern=end_matter_pattern,
    )


# --------------------------------------------------------------------------- #
# LevelSpec
# --------------------------------------------------------------------------- #


class TestLevelSpecValid:
    def test_pattern_is_compiled(self) -> None:
        spec = level("dieu", 2, DIEU)
        match = spec._regex.match("Điều 5. Phạm vi điều chỉnh")
        assert match is not None
        assert match.group("number") == "5"

    def test_rank_zero_allowed(self) -> None:
        assert level("chuong", 0, CHUONG).rank == 0

    @pytest.mark.parametrize(
        ("pattern", "text", "number"),
        [
            (CHUONG, "Chương III", "III"),
            (MUC, "Mục 2", "2"),
            (DIEU, "Điều 12: Hiệu lực", "12"),
            (KHOAN, "3. Đối với", "3"),
            (DIEM, "đ) Trường hợp", "đ"),
        ],
    )
    def test_realistic_patterns_accepted(self, pattern: str, text: str, number: str) -> None:
        match = level("x", 0, pattern)._regex.match(text)
        assert match is not None
        assert match.group("number") == number

    def test_compiled_pattern_not_in_repr(self) -> None:
        assert "_regex" not in repr(level("dieu", 2))

    def test_compiled_pattern_not_an_init_argument(self) -> None:
        with pytest.raises(TypeError):
            LevelSpec(  # type: ignore[call-arg]
                name="dieu",
                rank=2,
                pattern=DIEU,
                title_on_next_block=False,
                required_ancestor=None,
                _regex=re.compile(DIEU),
            )

    def test_equal_specs_compare_equal(self) -> None:
        assert level("dieu", 2) == level("dieu", 2)

    def test_frozen(self) -> None:
        spec = level("dieu", 2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.rank = 3  # type: ignore[misc]

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            LevelSpec("dieu", 2, DIEU, False, None)  # type: ignore[call-arg]


class TestLevelSpecInvalid:
    @pytest.mark.parametrize("name", ["", " ", "\t​"])
    def test_invisible_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="invisible"):
            level(name, 0)

    def test_negative_rank(self) -> None:
        with pytest.raises(ValueError, match=r"'dieu'.*rank >= 0, got -1"):
            level("dieu", -1)

    def test_broken_regex_names_the_level(self) -> None:
        with pytest.raises(ValueError, match=r"'dieu'.*failed to compile") as exc:
            level("dieu", 2, r"Điều\s+(?P<number>\d+")
        assert isinstance(exc.value.__cause__, re.error)

    @pytest.mark.parametrize(
        "pattern",
        [r"Điều\s+(\d+)\.", r"Điều\s+(?P<num>\d+)\."],
        ids=["unnamed-group", "wrong-group-name"],
    )
    def test_missing_number_group(self, pattern: str) -> None:
        with pytest.raises(ValueError, match="number group"):
            level("dieu", 2, pattern)

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(?P<number>)",
            r"(?P<number>\d*)",
            r"(Điều\s+)?(?P<number>\d*)",
            r"(?P<number>[a-zđ]?)\)?",
            r"(?P<number>.*)",
        ],
        ids=["empty-group", "digits-star", "optional-keyword", "optional-point", "dot-star"],
    )
    def test_pattern_matching_empty_text(self, pattern: str) -> None:
        with pytest.raises(ValueError, match="must not match empty string"):
            level("x", 0, pattern)

    def test_empty_pattern(self) -> None:
        with pytest.raises(ValueError):
            level("x", 0, "")


# --------------------------------------------------------------------------- #
# Profile: valid
# --------------------------------------------------------------------------- #


class TestProfileValid:
    def test_full_profile(self) -> None:
        p = profile()
        assert [lv.name for lv in p.levels or ()] == ["chuong", "muc", "dieu", "khoan", "diem"]

    def test_get_level(self) -> None:
        p = profile()
        assert p.get_level("khoan").rank == 3
        assert p.get_level("khoan").required_ancestor == "dieu"

    def test_get_level_missing(self) -> None:
        with pytest.raises(KeyError, match="'vn_regulation'.*'phan'"):
            profile().get_level("phan")

    def test_no_levels(self) -> None:
        p = profile(use_default_levels=False)
        assert p.levels is None
        with pytest.raises(KeyError):
            p.get_level("dieu")

    def test_no_end_matter(self) -> None:
        p = profile(end_matter_pattern=None)
        assert p._em_regex is None

    def test_no_levels_and_no_end_matter(self) -> None:
        p = profile(use_default_levels=False, end_matter_pattern=None)
        assert p.levels is None

    def test_end_matter_pattern_compiled(self) -> None:
        regex = profile()._em_regex
        assert regex is not None
        assert regex.match("Nơi nhận: - Như Điều 26") is not None

    def test_single_level(self) -> None:
        p = profile((level("dieu", 0),))
        assert p.get_level("dieu").rank == 0

    def test_ranks_may_have_gaps(self) -> None:
        p = profile((level("chuong", 0, CHUONG), level("dieu", 5), level("khoan", 9, KHOAN)))
        assert p.get_level("khoan").rank == 9

    def test_ancestor_two_levels_up(self) -> None:
        # The ancestor does not have to be the next rank up: any lower rank is valid.
        p = profile(
            (
                level("chuong", 0, CHUONG),
                level("diem", 4, DIEM, required_ancestor="chuong"),
            )
        )
        assert p.get_level("diem").required_ancestor == "chuong"

    def test_mapping_is_read_only(self) -> None:
        p = profile()
        with pytest.raises(TypeError):
            p._level_by_name["x"] = level("x", 9)  # type: ignore[index]

    def test_internal_fields_not_in_repr(self) -> None:
        text = repr(profile())
        assert "_level_by_name" not in text
        assert "_em_regex" not in text

    def test_equal_profiles_compare_equal(self) -> None:
        assert profile() == profile()

    def test_frozen(self) -> None:
        p = profile()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.name = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Profile: invalid
# --------------------------------------------------------------------------- #


class TestProfileInvalid:
    @pytest.mark.parametrize(("name", "version"), [("", "1.0.0"), (" ", "1.0.0"), ("p", ""), ("p", "\t")])
    def test_invisible_name_or_version(self, name: str, version: str) -> None:
        with pytest.raises(ValueError, match="invisible"):
            profile(name=name, version=version)

    def test_empty_levels_tuple(self) -> None:
        with pytest.raises(ValueError, match="None"):
            profile(())

    def test_duplicate_level_name(self) -> None:
        with pytest.raises(ValueError, match="duplicate level 'dieu'"):
            profile((level("dieu", 0), level("dieu", 1)))

    def test_duplicate_rank_names_both_levels(self) -> None:
        with pytest.raises(ValueError, match="duplicate rank between levels 'khoan' and 'dieu'"):
            profile((level("dieu", 2), level("khoan", 2, KHOAN)))

    def test_levels_not_sorted(self) -> None:
        with pytest.raises(ValueError, match="sorted by rank"):
            profile((level("dieu", 2), level("chuong", 0, CHUONG)))

    def test_unknown_ancestor(self) -> None:
        with pytest.raises(ValueError, match=r"'khoan'.*'dieuu'.*non-existing"):
            profile((level("dieu", 2), level("khoan", 3, KHOAN, required_ancestor="dieuu")))

    def test_ancestor_with_higher_rank(self) -> None:
        with pytest.raises(ValueError, match=r"'khoan'.*invalid ancestor 'diem'"):
            profile(
                (
                    level("dieu", 2),
                    level("khoan", 3, KHOAN, required_ancestor="diem"),
                    level("diem", 4, DIEM),
                )
            )

    def test_level_requiring_itself(self) -> None:
        with pytest.raises(ValueError, match=r"'dieu'.*invalid ancestor 'dieu'"):
            profile((level("dieu", 2, required_ancestor="dieu"),))

    def test_broken_end_matter_pattern(self) -> None:
        with pytest.raises(ValueError, match="failed to compile end_matter_pattern") as exc:
            profile(end_matter_pattern=r"Nơi nhận(")
        assert isinstance(exc.value.__cause__, re.error)

    @pytest.mark.parametrize("pattern", ["", r"\s*", r"(Nơi nhận)?", ".*"])
    def test_end_matter_pattern_matching_empty_text(self, pattern: str) -> None:
        with pytest.raises(ValueError, match="must not match empty string"):
            profile(end_matter_pattern=pattern)

    def test_broken_end_matter_reported_without_levels(self) -> None:
        # The error message must not depend on any level existing.
        with pytest.raises(ValueError, match="failed to compile"):
            profile(use_default_levels=False, end_matter_pattern="(")