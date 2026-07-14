"""Tests for :func:`radlabels._validation.validate_aliases`."""
from __future__ import annotations

from radlabels._validation import validate_aliases
from radlabels.aliases import ALIASES


# ------------------------------------------------------------------ #
#                        schema error cases                          #
# ------------------------------------------------------------------ #
def test_non_dict_entry_is_error():
    bad = {"cardiomegaly": ["cardiomegaly"]}  # list instead of dict
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert errors, "expected an error for non-dict entry"


def test_missing_aliases_key_is_error():
    bad = {"cardiomegaly": {"exclude": []}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert any("missing" in e and "aliases" in e for e in errors)


def test_empty_aliases_list_is_error():
    bad = {"cardiomegaly": {"aliases": [], "exclude": []}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert any("at least one phrase" in e for e in errors)


def test_blank_phrase_is_error():
    bad = {"cardiomegaly": {"aliases": ["cardiomegaly", "  "], "exclude": []}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert errors


def test_non_string_phrase_is_error():
    bad = {"cardiomegaly": {"aliases": ["cardiomegaly", 42], "exclude": []}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert errors


def test_non_list_exclude_is_error():
    bad = {"cardiomegaly": {"aliases": ["cardiomegaly"], "exclude": "pericardial effusion"}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert any("exclude" in error and "must be a list" in error for error in errors)


def test_invalid_exclude_phrase_is_error():
    bad = {"cardiomegaly": {"aliases": ["cardiomegaly"], "exclude": ["", 42]}}
    msgs = validate_aliases(bad)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert len([error for error in errors if ".exclude[" in error]) == 2


# ------------------------------------------------------------------ #
#                      duplicate-phrase warnings                     #
# ------------------------------------------------------------------ #
def test_duplicate_phrase_across_labels_is_warning():
    aliases = {
        "label_a": {"aliases": ["pleural effusion", "fluid"], "exclude": []},
        "label_b": {"aliases": ["pleural effusion", "pericardial effusion"], "exclude": []},
    }
    msgs = validate_aliases(aliases)
    warnings = [m for m in msgs if m.startswith("WARNING")]
    assert any("pleural effusion" in w for w in warnings)


def test_no_duplicate_within_single_label():
    aliases = {
        "label_a": {"aliases": ["pleural effusion", "hydrothorax"], "exclude": []},
        "label_b": {"aliases": ["pericardial effusion"], "exclude": []},
    }
    msgs = validate_aliases(aliases)
    warnings = [m for m in msgs if m.startswith("WARNING")]
    assert not warnings


# ------------------------------------------------------------------ #
#                      built-in dictionary sanity                    #
# ------------------------------------------------------------------ #
def test_builtin_aliases_pass_schema():
    """Built-in ALIASES must have no schema errors."""
    msgs = validate_aliases(ALIASES)
    errors = [m for m in msgs if m.startswith("ERROR")]
    assert not errors, f"built-in ALIASES has schema errors: {errors}"


def test_builtin_aliases_returns_list():
    result = validate_aliases(ALIASES)
    assert isinstance(result, list)
