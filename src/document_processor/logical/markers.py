import re
from typing import Literal

type MatchSource = Literal["text", "list_marker"]
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
WORD_BULLET = "word_bullet"
