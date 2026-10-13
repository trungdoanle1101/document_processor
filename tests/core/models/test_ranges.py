"""Tests for TextRange."""

import dataclasses

import pytest

from document_processor.core.models.ranges import TextRange

TEXT = "Điều 1. Phạm vi điều chỉnh"


class TestTextRangeValidation:
    @pytest.mark.parametrize(("start", "end"), [(0, 1), (0, 26), (8, 26), (5, 6)])
    def test_valid(self, start: int, end: int) -> None:
        r = TextRange(start, end)
        assert (r.start, r.end) == (start, end)

    @pytest.mark.parametrize(
        ("start", "end"),
        [(-1, 5), (3, 3), (0, 0), (5, 2)],
        ids=["negative-start", "empty", "empty-at-zero", "reversed"],
    )
    def test_invalid(self, start: int, end: int) -> None:
        with pytest.raises(ValueError, match="0 <= start < end"):
            TextRange(start, end)

    def test_frozen(self) -> None:
        r = TextRange(0, 1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.start = 1  # type: ignore[misc]


class TestTextRangeSlice:
    def test_marker(self) -> None:
        assert TextRange(0, 7).slice(TEXT) == "Điều 1."

    def test_title(self) -> None:
        assert TextRange(8, 26).slice(TEXT) == "Phạm vi điều chỉnh"

    def test_whole_text(self) -> None:
        assert TextRange(0, len(TEXT)).slice(TEXT) == TEXT

    def test_end_equal_to_length_is_allowed(self) -> None:
        assert TextRange(len(TEXT) - 1, len(TEXT)).slice(TEXT) == "h"

    def test_end_past_text_raises(self) -> None:
        with pytest.raises(ValueError, match="end 27 > len 26"):
            TextRange(0, 27).slice(TEXT)

    def test_counts_code_points_not_bytes(self) -> None:
        # "Đ" is 2 bytes in UTF-8 but one code point.
        assert TextRange(0, 1).slice("Điều") == "Đ"