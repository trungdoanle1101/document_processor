import re
from dataclasses import dataclass, field
from typing import Mapping
from types import MappingProxyType

from document_processor.core.text_utils import is_invisible


@dataclass(frozen=True, kw_only=True)
class LevelSpec:
    # Name of the level
    name: str

    # Rank (0-based) of the level within a profile's hierarchy
    rank: int

    # Pattern string that will be compiled post-init to find the pattern within a document
    pattern: str

    # Some levels have their title displayed on the next block rather than the same line
    title_on_next_block: bool    
    
    # required_ancestor refers to another level in the Profile whose rank < self rank:
    # any ancestor, not only direct parent
    # only requires levels that are always present
    required_ancestor: str | None

    # Private compiled pattern from 'pattern' attribute
    _regex: re.Pattern[str] = field(init=False, repr=False, compare=False)


    def __post_init__(self) -> None:
        if is_invisible(self.name):
            raise ValueError(f"LevelSpec {self.name!r}: must not contain all invisible characters")
        
        if self.rank < 0:
            raise ValueError(f"LevelSpec {self.name!r}: required: rank >= 0, got {self.rank}")

        try:
            regex = re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"LevelSpec {self.name!r}: failed to compile pattern {self.pattern!r}") from e
        
        if "number" not in regex.groupindex:
            raise ValueError(f"LevelSpec {self.name!r}: pattern {self.pattern!r} does not contain a number group (?P<number>)")
        
        # Check faulty regex that matches empty
        if regex.match("") is not None:
            raise ValueError(f"LevelSpec {self.name!r}: pattern {self.pattern!r} must not match empty string")
        
        object.__setattr__(self, "_regex", regex)


@dataclass(frozen=True, kw_only=True)
class Profile:
    name: str
    version: str
    levels: tuple[LevelSpec, ...] | None
    
    # Pattern string that will be compiled post-init to find the beginning of end_matter
    end_matter_pattern: str | None

    # Mapping to get level by name
    _level_by_name: Mapping[str, LevelSpec] = field(init=False, repr=False, compare=False)

    # Private compiled pattern from 'end_matter_pattern' attribute
    _em_regex: re.Pattern[str] | None = field(init=False, repr=False, compare=False)


    def __post_init__(self) -> None:

        if is_invisible(self.name) or is_invisible(self.version):
            raise ValueError(f"Profile {self.name!r}: must not contain all invisible characters for name and version")
        
        try:
            em_regex = re.compile(self.end_matter_pattern) if self.end_matter_pattern is not None else None
        except re.error as e:
            raise ValueError(f"Profile {self.name!r}: failed to compile end_matter_pattern {self.end_matter_pattern!r}") from e
        
        if em_regex is not None and em_regex.match("") is not None:
            raise ValueError(f"Profile {self.name!r}: end_matter_regex {self.end_matter_pattern!r} must not match empty string")
        
        level_by_name: dict[str, LevelSpec] = dict()
        if self.levels is not None:
            # Check empty levels provided
            if len(self.levels) == 0:
                raise ValueError(f"Profile {self.name!r}: Defined levels cannot be an empty tuple. Hint: Provide level or assign None explicitly if there's no level hierarchy")
            
            # Check if levels are sorted by rank
            for i in range(len(self.levels) - 1):
                if self.levels[i].rank > self.levels[i+1].rank:
                    raise ValueError(f"Profile {self.name!r}: levels in a profile must be sorted by rank in ascending order")

            name_by_rank: dict[int, str] = dict()
            for level in self.levels:
                # Check unique levels
                if level.name in level_by_name:
                    raise ValueError(f"Profile {self.name!r}: duplicate level {level.name!r}")
                level_by_name[level.name] = level

                # Check for duplicate ranks
                if level.rank in name_by_rank:
                    raise ValueError(f"Profile {self.name!r}: duplicate rank between levels {level.name!r} and {name_by_rank[level.rank]!r}")
                name_by_rank[level.rank] = level.name

            for level in self.levels:
                # Check required_ancestor
                if level.required_ancestor is not None:
                    if level.required_ancestor not in level_by_name:
                        raise ValueError(f"Profile {self.name!r}, level {level.name!r}: invalid ancestor {level.required_ancestor!r}: refers to a non-existing level")
                    
                    ancestor_level = level_by_name[level.required_ancestor]
                    if level.rank <= ancestor_level.rank:
                        raise ValueError(f"Profile {self.name!r}, level {level.name!r}: invalid ancestor {level.required_ancestor!r}: level's rank ({level.rank}) precedes or is indentical to ancestor's rank ({ancestor_level.rank})")
                    

        object.__setattr__(self, "_level_by_name", MappingProxyType(level_by_name))
        object.__setattr__(self, "_em_regex", em_regex)

    def get_level(self, name: str) -> LevelSpec:
        if name not in self._level_by_name:
            raise KeyError(f"Profile {self.name!r}: does not contain level {name!r}")
        return self._level_by_name[name]

        



