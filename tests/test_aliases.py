"""Sanity checks on the alias dictionary."""
from __future__ import annotations

import re

from radlabels.aliases import ALIAS_VERSION, ALIASES, LABEL_NAMES, PARENT_MAP

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


# ------------------------------------------------------------------ #
#                         PARENT_MAP tests                           #
# ------------------------------------------------------------------ #
_EXPECTED_PARENTS = {
    "pulmonary_abnormality",
    "pleural_abnormality",
    "cardiomediastinal_abnormality",
    "airway_abnormality",
    "hilar_abnormality",
    "fracture_or_trauma",
    "mediastinal_or_abdominal_air",
    "hernia_abnormality",
    "support_devices",
}


def test_parent_map_covers_all_aliases():
    """Every key in ALIASES must appear in PARENT_MAP."""
    unmapped = set(ALIASES.keys()) - set(PARENT_MAP.keys())
    assert not unmapped, f"PARENT_MAP is missing leaf labels: {sorted(unmapped)}"


def test_parent_map_no_extra_leaves():
    """PARENT_MAP must not reference labels absent from ALIASES."""
    extra = set(PARENT_MAP.keys()) - set(ALIASES.keys())
    assert not extra, f"PARENT_MAP contains unknown leaf labels: {sorted(extra)}"


def test_parent_map_parent_names():
    """All parent values must be one of the nine documented coarse categories."""
    unexpected = set(PARENT_MAP.values()) - _EXPECTED_PARENTS
    assert not unexpected, f"PARENT_MAP has unexpected parent names: {sorted(unexpected)}"


def test_parent_map_disputed_labels():
    """consolidation, ground_glass_opacity, and pneumonia must map to pulmonary_abnormality."""
    for label in ("consolidation", "ground_glass_opacity", "pneumonia"):
        assert PARENT_MAP.get(label) == "pulmonary_abnormality", (
            f"{label!r} should map to 'pulmonary_abnormality', "
            f"got {PARENT_MAP.get(label)!r}"
        )
