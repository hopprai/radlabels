"""Sanity checks on the alias dictionary."""
from __future__ import annotations

import re

from radlabels.aliases import ALIAS_VERSION, ALIASES, LABEL_NAMES

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_alias_version_is_string():
    assert isinstance(ALIAS_VERSION, str)
    assert ALIAS_VERSION


def test_label_names_sorted_and_unique():
    assert LABEL_NAMES == sorted(set(ALIASES.keys()))


def test_label_keys_are_snake_case():
    for key in ALIASES:
        assert _KEY_RE.match(key), f"label key {key!r} is not snake_case"


def test_every_label_has_at_least_one_alias():
    for key, entry in ALIASES.items():
        assert isinstance(entry, dict), f"{key}: entry must be dict"
        aliases = entry.get("aliases", [])
        assert isinstance(aliases, list), f"{key}: aliases must be a list"
        assert len(aliases) >= 1, f"{key}: must have at least 1 alias phrase"
        for phrase in aliases:
            assert isinstance(phrase, str) and phrase.strip(), \
                f"{key}: alias phrase must be a non-empty string, got {phrase!r}"


def test_exclude_field_optional_but_well_typed():
    for key, entry in ALIASES.items():
        excl = entry.get("exclude", [])
        assert isinstance(excl, list)
        for phrase in excl:
            assert isinstance(phrase, str) and phrase.strip()


def test_no_unicode_bombs():
    for key, entry in ALIASES.items():
        for phrase in entry["aliases"] + entry.get("exclude", []):
            phrase.encode("ascii")  # raises UnicodeEncodeError if not ascii
