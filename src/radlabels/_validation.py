"""Alias dictionary validation utilities."""
from __future__ import annotations

from .matcher import normalize_tokens


def validate_aliases(aliases: dict) -> list[str]:
    """Validate an alias dictionary, returning a list of error/warning strings.

    Errors (prefix ``ERROR:``) are returned first; callers should fail fast
    if any errors are present.  Warnings (prefix ``WARNING:``) follow and are
    informational — they describe overlaps that may be intentional.

    Parameters
    ----------
    aliases
        A dict following the ``ALIASES`` schema::

            {label_key: {"aliases": [phrase, ...], "exclude": [phrase, ...]}}

    Returns
    -------
    list[str]
        Empty list means the dictionary is clean.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ---- schema checks (errors) ------------------------------------------
    for key, entry in aliases.items():
        if not isinstance(entry, dict):
            errors.append(f"ERROR: '{key}' entry must be a dict, got {type(entry).__name__}")
            continue
        alias_list = entry.get("aliases")
        if alias_list is None:
            errors.append(f"ERROR: '{key}' is missing required key 'aliases'")
            continue
        if not isinstance(alias_list, list):
            errors.append(f"ERROR: '{key}.aliases' must be a list, got {type(alias_list).__name__}")
            continue
        if len(alias_list) == 0:
            errors.append(f"ERROR: '{key}.aliases' must have at least one phrase")
            continue
        for i, phrase in enumerate(alias_list):
            if not isinstance(phrase, str) or not phrase.strip():
                errors.append(
                    f"ERROR: '{key}.aliases[{i}]' must be a non-empty string, got {phrase!r}"
                )

    # ---- duplicate-phrase check (warnings) --------------------------------
    # Build normalized-phrase → [label, ...] mapping.
    phrase_to_labels: dict[str, list[str]] = {}
    for key, entry in aliases.items():
        if not isinstance(entry, dict):
            continue
        for phrase in entry.get("aliases", []):
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            normed = " ".join(normalize_tokens(phrase))
            phrase_to_labels.setdefault(normed, []).append(key)

    for normed_phrase, labels in phrase_to_labels.items():
        if len(labels) > 1:
            warnings.append(
                f"WARNING: phrase '{normed_phrase}' appears in multiple labels: "
                + ", ".join(sorted(set(labels)))
            )

    return errors + warnings
