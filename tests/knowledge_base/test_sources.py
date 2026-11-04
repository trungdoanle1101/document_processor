"""Cited sources in a reply are shown short, by rules the answer style chooses."""

from dataclasses import replace

import pytest

from knowledge_base.assistant import short_source
from knowledge_base.templates import GENERAL_EN, VN_REGULATION

TITLE = "Quy định quản lý kiến trúc CNTT"
STAGING = (
    "Chương III KIẾN TRÚC CNTT THÀNH PHẦN",
    "Mục III KIẾN TRÚC DỮ LIỆU",
    "Điều 9. Mô hình kiến trúc dữ liệu",
    "2. Mô tả các thành phần",
    "2.3. Tầng xử lý và lưu trữ dữ liệu (Data Processing and Stores):",
    "2.3.2 Tầng lưu trữ dữ liệu (Data Stores):",
    "a. Vùng hồ dữ liệu (Data Lakehouse): Được tổ chức thành các phân vùng lưu trữ sau:",
    "-",
)


def test_a_deep_regulation_passage_keeps_its_article_and_last_headings() -> None:
    assert short_source(TITLE, STAGING, VN_REGULATION) == (
        "Quy định quản lý kiến trúc CNTT › Điều 9. Mô hình kiến trúc dữ liệu › … "
        "› 2.3.2 Tầng lưu trữ dữ liệu (Data Stores) › a. Vùng hồ dữ liệu (Data Lakehouse)"
    )


def test_without_an_anchor_only_the_last_headings_show() -> None:
    assert short_source(TITLE, STAGING, GENERAL_EN) == (
        "Quy định quản lý kiến trúc CNTT "
        "› 2.3.2 Tầng lưu trữ dữ liệu (Data Stores) › a. Vùng hồ dữ liệu (Data Lakehouse)"
    )


def test_a_shallow_passage_shows_every_heading_without_a_gap() -> None:
    headings = ("Chương I QUY ĐỊNH CHUNG", "Điều 2. Đối tượng áp dụng")
    assert short_source(TITLE, headings, VN_REGULATION) == (
        "Quy định quản lý kiến trúc CNTT › Chương I QUY ĐỊNH CHUNG › Điều 2. Đối tượng áp dụng"
    )


def test_an_anchor_among_the_last_headings_is_not_repeated() -> None:
    headings = ("Chương II", "Điều 5. Hiệu lực", "1. Quy định có hiệu lực từ ngày ký")
    assert short_source(TITLE, headings, VN_REGULATION) == (
        "Quy định quản lý kiến trúc CNTT › Điều 5. Hiệu lực › 1. Quy định có hiệu lực từ ngày ký"
    )


def test_no_headings_is_the_title_alone() -> None:
    assert short_source(TITLE, (), VN_REGULATION) == TITLE


def test_a_short_heading_with_a_colon_is_kept_whole() -> None:
    style = replace(GENERAL_EN, source_depth=1)
    assert (
        short_source(TITLE, ("Article 5: Retention",), style)
        == f"{TITLE} › Article 5: Retention"
    )


def test_a_long_heading_without_a_colon_is_cut_at_a_word() -> None:
    style = replace(GENERAL_EN, source_depth=1, source_heading_chars=20)
    long = "Dữ liệu được lưu trữ ngắn hạn tối đa bảy ngày"
    assert short_source(TITLE, (long,), style) == f"{TITLE} › Dữ liệu được lưu trữ…"


def test_depth_zero_with_an_anchor_shows_the_article_alone() -> None:
    style = replace(VN_REGULATION, source_depth=0)
    assert (
        short_source(TITLE, STAGING, style)
        == f"{TITLE} › Điều 9. Mô hình kiến trúc dữ liệu"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_depth": -1}, "source_depth"),
        ({"source_heading_chars": 5}, "source_heading_chars"),
        ({"source_anchor": "(unclosed"}, "missing"),
    ],
)
def test_bad_style_settings_are_refused(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(Exception, match=message):
        replace(VN_REGULATION, **changes)  # type: ignore[arg-type]