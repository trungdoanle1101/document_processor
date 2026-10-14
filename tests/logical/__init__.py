from document_processor.logical.profile import Profile, LevelSpec

# Example: Chương III
CHUONG = r"Chương\s+(?P<number>[IVXLC]+)\b"

# Example: Mục II or Mục 2
MUC = r"Mục\s+(?P<number>[IVXLC]+|\d+)\b"

# Example: Điều 10. or Điều 10:
DIEU = r"Điều\s+(?P<number>\d+)[.:]"

KHOAN = r"(?P<number>\d+)\.(?!\d)"

DIEM = r"(?P<number>[a-zđ]+)\)"

END_MATTER = "Nơi nhận"

VN_REGULATION = Profile(
    name="vn_regulation",
    version="1",
    levels=(
        LevelSpec(name="chuong", rank=0, pattern=CHUONG, title_on_next_block=True, required_ancestor=None),
        LevelSpec(name="muc", rank=1, pattern=MUC, title_on_next_block=True, required_ancestor=None),
        LevelSpec(name="dieu", rank=2, pattern=DIEU, title_on_next_block=False, required_ancestor=None),
        LevelSpec(name="khoan", rank=3, pattern=KHOAN, title_on_next_block=False, required_ancestor="dieu"),
        LevelSpec(name="diem", rank=4, pattern=DIEM, title_on_next_block=False, required_ancestor="dieu")
    ),
    end_matter_pattern=END_MATTER
)