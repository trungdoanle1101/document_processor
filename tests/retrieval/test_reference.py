"""address_keys: every way the nodes on a chunk's path can be cited.

Citations usually leave out the outer levels ("khoản 2 Điều 14", not "... Chương II"), so
each ancestor gets its full address and every shorter ending of it.
"""

from document_processor.chunking import PathStep
from document_processor.retrieval.reference import address_keys


def step(level: str, number: str | None, start: int) -> PathStep:
    marker = None if number is None else f"{level} {number}"
    return PathStep(
        node_id=f"n{start}",
        level=level,
        number=number,
        marker=marker,
        title=None,
        start=start,
    )


CHUONG_II = step("chuong", "II", 0)
MUC_1 = step("muc", "1", 1)
DIEU_14 = step("dieu", "14", 2)
KHOAN_2 = step("khoan", "2", 3)
DIEM_A = step("diem", "a", 4)
GACH = step("gach", None, 5)


def test_each_ancestor_gets_its_full_address_and_every_shorter_ending() -> None:
    assert address_keys((CHUONG_II, DIEU_14, KHOAN_2)) == (
        "chuong:ii",
        "chuong:ii/dieu:14",
        "dieu:14",
        "chuong:ii/dieu:14/khoan:2",
        "dieu:14/khoan:2",
        "khoan:2",
    )


def test_the_usual_citations_all_match() -> None:
    keys = address_keys((CHUONG_II, MUC_1, DIEU_14, KHOAN_2, DIEM_A))
    for cited in (
        "chuong:ii",  # Chương II
        "chuong:ii/muc:1",  # Mục 1 Chương II
        "dieu:14",  # Điều 14
        "dieu:14/khoan:2",  # khoản 2 Điều 14
        "dieu:14/khoan:2/diem:a",  # điểm a khoản 2 Điều 14
    ):
        assert cited in keys


def test_an_article_keeps_its_own_key_under_a_section() -> None:
    keys = address_keys((CHUONG_II, MUC_1, DIEU_14))
    assert "dieu:14" in keys
    assert "chuong:ii/muc:1/dieu:14" in keys


def test_a_sibling_is_not_matched() -> None:
    keys = address_keys((DIEU_14, KHOAN_2))
    assert "dieu:14/khoan:3" not in keys
    assert "dieu:15" not in keys


def test_numbers_are_case_folded() -> None:
    assert address_keys((step("dieu", "5", 0), step("diem", "A", 1))) == (
        "dieu:5",
        "dieu:5/diem:a",
        "diem:a",
    )


def test_numbers_are_kept_as_written_otherwise() -> None:
    assert "dieu:5/khoan_cap_2:2.1" in address_keys(
        (step("dieu", "5", 0), step("khoan_cap_2", "2.1", 1))
    )


def test_keys_stop_at_the_first_step_without_a_number() -> None:
    """Nothing below a bullet can be cited without naming the bullet, which has no number."""
    below = step("khoan", "1", 6)
    assert address_keys((DIEU_14, KHOAN_2, GACH, below)) == address_keys(
        (DIEU_14, KHOAN_2)
    )


def test_a_path_starting_without_a_number_has_no_keys() -> None:
    assert address_keys((GACH, KHOAN_2)) == ()


def test_an_empty_path_has_no_keys() -> None:
    """Front and end matter chunks have no path."""
    assert address_keys(()) == ()


def test_each_key_appears_once() -> None:
    keys = address_keys((step("muc", "1", 0), step("muc", "1", 1)))
    assert len(keys) == len(set(keys))
    assert keys == ("muc:1", "muc:1/muc:1")
