"""Match radiologist-defined alias phrases against RadGraph annotations.

Public entry point: :func:`label_study`. Given a single RadGraph annotation
for one report, returns a dict of ``{label: status}`` plus a list of
per-alias matches with token positions for traceability.

The matcher does no training and no LLM calls. It walks the RadGraph
relational neighborhood around each Observation entity, normalizes tokens
(lowercase + WordNet lemma), and tests whether any alias phrase's tokens
form a subset of the neighborhood's bag of tokens.

See :mod:`radlabels.aliases` for the dictionary schema.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .aliases import ALIASES

# ------------------------------------------------------------------ #
#                          NLTK BOOTSTRAP                            #
# ------------------------------------------------------------------ #
# WordNet lemmatization is used to normalize both report tokens and alias
# phrases so that "fractures" matches "fracture", etc.
import nltk  # noqa: E402

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass

from nltk.stem import WordNetLemmatizer  # noqa: E402

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1_000_000)
def _lemmatize_cached(tok: str) -> str:
    return _lemmatizer.lemmatize(tok)


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, whitespace-split, and lemmatize tokens.

    The same function is applied to alias phrases and report-derived
    tokens so the match space is consistent.
    """
    return [_lemmatize_cached(t) for t in text.lower().split()]


# ------------------------------------------------------------------ #
#                    STATIC LOOKUP TABLES (precomputed)              #
# ------------------------------------------------------------------ #
LABEL_NAMES: list[str] = sorted(ALIASES.keys())
PRETTY: dict[str, str] = {d: d.replace("_", " ") for d in LABEL_NAMES}


def _build_phrase_tokensets(
    phrases: Iterable[str],
) -> tuple[list[str], list[list[str]], list[set[str]]]:
    """Normalize a list of phrases into (canonical_strings, token_lists, token_sets)."""
    normed = sorted({" ".join(normalize_tokens(p)) for p in phrases if p} - {""})
    toklists = [normalize_tokens(p) for p in normed]
    toksets = [set(tl) for tl in toklists]
    return normed, toklists, toksets


# Per-label precomputed phrase tables. The canonical "pretty" form is
# included as a default alias if the label has at least one explicit alias.
_ALL_PHRASES: dict[str, list[str]] = {}
_ALIAS_TOKEN_LISTS: dict[str, list[list[str]]] = {}
_ALIAS_TOKEN_SETS: dict[str, list[set[str]]] = {}
_EXCLUDE_TOKEN_SETS: dict[str, list[set[str]]] = {}

for _dz in LABEL_NAMES:
    _spec = ALIASES[_dz]
    _base = list(_spec.get("aliases", []))
    if _base:
        _base = [PRETTY[_dz]] + _base
    _phrases, _toklists, _toksets = _build_phrase_tokensets(_base)
    _ALL_PHRASES[_dz] = _phrases
    _ALIAS_TOKEN_LISTS[_dz] = _toklists
    _ALIAS_TOKEN_SETS[_dz] = _toksets

    _, _, _excl = _build_phrase_tokensets(_spec.get("exclude", []))
    _EXCLUDE_TOKEN_SETS[_dz] = _excl

del _dz, _spec, _base, _phrases, _toklists, _toksets, _excl

# ------------------------------------------------------------------ #
#                  RADGRAPH STATUS NORMALIZATION                     #
# ------------------------------------------------------------------ #
_RG_PREFIX = "Observation::"
_OBS_MAP = {
    "present": "definitely present",
    "definitely present": "definitely present",
    "absent": "definitely absent",
    "definitely absent": "definitely absent",
    "uncertain": "uncertain",
}
_MODIFIER_REL = "modify"
_STATUS_PRIORITY = ("definitely present", "uncertain", "definitely absent")
_VALID_UNCERTAINTY_POLICIES = frozenset({"keep", "as_positive", "as_negative", "drop"})


def canonical_status(rg_label: str, relations: list[tuple[str, str]]) -> str | None:
    """Return canonical presence status for an Observation seed entity.

    Returns ``None`` for non-Observation entities, modifier-only entities
    (those with an outgoing ``modify`` relation, e.g. "stable", "no"), and
    unknown status strings.
    """
    if not rg_label.startswith(_RG_PREFIX):
        return None
    for rel_type, _ in relations or []:
        if (rel_type or "").lower().strip() == _MODIFIER_REL:
            return None
    return _OBS_MAP.get(rg_label[len(_RG_PREFIX):].strip().lower())


def _collapse_statuses(statuses: Iterable[str]) -> str:
    """Priority-collapse a bag of statuses to a single label."""
    s = set(statuses)
    for p in _STATUS_PRIORITY:
        if p in s:
            return p
    return "definitely absent"


# ------------------------------------------------------------------ #
#                 ENTITY EXPANSION + NEIGHBORHOOD WALK               #
# ------------------------------------------------------------------ #
def _explode_entity_tokens(ent: dict) -> list[tuple[str, int]]:
    """Convert ``{"tokens": "w1 w2", "start_ix": k, ...}`` to ``[(w1, k), (w2, k+1)]``."""
    words = normalize_tokens(ent.get("tokens", ""))
    start_ix = int(ent.get("start_ix", 0))
    return [(w, start_ix + i) for i, w in enumerate(words)]


def _candidate_token_map(
    ent_id: str, entities: dict
) -> tuple[set[str], dict[str, list[int]]]:
    """Build the relational neighborhood around a seed entity.

    Walk all OUTGOING relations transitively, plus include any entity with
    a one-hop INCOMING relation to the seed and expand those outgoing too.
    Return the union of normalized tokens plus a token \u2192 positions map.
    """
    visited: set[str] = set()
    token2ix: dict[str, list[int]] = defaultdict(list)

    def collect(entity_id: str) -> None:
        if entity_id in visited or entity_id not in entities:
            return
        visited.add(entity_id)
        ent = entities[entity_id]
        for tok, ix in _explode_entity_tokens(ent):
            token2ix[tok].append(ix)
        for _, rel_id in ent.get("relations", []):
            collect(rel_id)

    collect(ent_id)

    for other_id, other_ent in entities.items():
        if other_id == ent_id:
            continue
        for _, rel_id in other_ent.get("relations", []):
            if rel_id == ent_id:
                collect(other_id)

    return set(token2ix.keys()), token2ix


def _indices_for_alias(
    token2ix: dict[str, list[int]], alias_tokens: list[str]
) -> list[int] | None:
    """Pick one token position per alias token, greedily preferring unused ones.

    Returns ``None`` if any alias token is missing from the neighborhood.
    The returned indices point into the report's whitespace-split text but
    are NOT guaranteed to be contiguous or in ascending order \u2014 matching
    is bag-of-words within a RadGraph cluster.
    """
    chosen: list[int] = []
    used: set[int] = set()
    for tok in alias_tokens:
        positions = token2ix.get(tok)
        if not positions:
            return None
        ix_list = sorted(positions)
        pick = next((ix for ix in ix_list if ix not in used), ix_list[0])
        chosen.append(pick)
        used.add(pick)
    return chosen


# ------------------------------------------------------------------ #
#                      COMPILED-ALIASES HELPER                       #
# ------------------------------------------------------------------ #
@dataclass
class _CompiledAliases:
    label_names: list[str]
    all_phrases: dict[str, list[str]]
    alias_toklists: dict[str, list[list[str]]]
    alias_toksets: dict[str, list[set[str]]]
    excl_toksets: dict[str, list[set[str]]]


def _compile_aliases(aliases: dict) -> _CompiledAliases:
    """Compile a custom alias dict into lookup tables (once per batch)."""
    label_names = sorted(aliases.keys())
    pretty = {d: d.replace("_", " ") for d in label_names}
    all_phrases: dict[str, list[str]] = {}
    alias_toklists: dict[str, list[list[str]]] = {}
    alias_toksets: dict[str, list[set[str]]] = {}
    excl_toksets: dict[str, list[set[str]]] = {}
    for dz in label_names:
        spec = aliases[dz]
        base = list(spec.get("aliases", []))
        if base:
            base = [pretty[dz]] + base
        phrases, toklists, toksets = _build_phrase_tokensets(base)
        all_phrases[dz] = phrases
        alias_toklists[dz] = toklists
        alias_toksets[dz] = toksets
        _, _, excl = _build_phrase_tokensets(spec.get("exclude", []))
        excl_toksets[dz] = excl
    return _CompiledAliases(
        label_names=label_names,
        all_phrases=all_phrases,
        alias_toklists=alias_toklists,
        alias_toksets=alias_toksets,
        excl_toksets=excl_toksets,
    )


# ------------------------------------------------------------------ #
#                          PUBLIC ENTRY POINT                        #
# ------------------------------------------------------------------ #
def label_study(
    rg_anno: dict,
    annotator_key: str = "0",
    *,
    aliases: dict | None = None,
    _compiled: "_CompiledAliases | None" = None,
    apply_exclude: bool = True,
    uncertainty_policy: str = "keep",
) -> tuple[dict[str, str], list[dict], str]:
    """Label a single RadGraph-annotated study.

    Parameters
    ----------
    rg_anno
        A single study's RadGraph annotation, shaped like::

            {"0": {"text": str,
                   "entities": {eid: {"tokens", "label", "start_ix",
                                       "end_ix", "relations"}}}}
    annotator_key
        Which annotator key to read from ``rg_anno``. Defaults to ``"0"``.
    aliases
        Custom alias dictionary following the ``ALIASES`` schema.  When
        provided, it **fully replaces** the built-in dictionary for this call.
        Pass ``None`` (default) to use the built-in dictionary.
    apply_exclude
        If True (default), an ``exclude`` phrase appearing in the same
        relational neighborhood as a seed vetoes the hit for that label.
    uncertainty_policy
        How to handle ``uncertain`` per-seed statuses. One of:

        - ``"keep"``     emit ``uncertain`` as a status (default).
        - ``"as_positive"`` re-map uncertain to ``definitely present``.
        - ``"as_negative"`` re-map uncertain to ``definitely absent``.
        - ``"drop"``     drop uncertain hits entirely.

    Returns
    -------
    labels : dict[str, str]
        ``{label: "definitely present" | "uncertain" | "definitely absent"}``
        for labels that had at least one alias match. Missing keys mean
        "no evidence found" \u2014 do NOT treat them as definitely absent.
    matches : list[dict]
        One entry per resolved alias hit::

            {"disease": str, "alias": str, "label": str, "start_ix": [int, ...]}
    text : str
        The pre-tokenized report text from RadGraph.
    """
    if uncertainty_policy not in _VALID_UNCERTAINTY_POLICIES:
        raise ValueError(
            f"uncertainty_policy must be one of {sorted(_VALID_UNCERTAINTY_POLICIES)!r}, "
            f"got {uncertainty_policy!r}"
        )

    sub = rg_anno.get(annotator_key) if isinstance(rg_anno, dict) else None
    if not sub or "entities" not in sub:
        return {}, [], ""

    text = sub.get("text", "") or ""
    entities = sub["entities"]

    # Resolve alias lookup tables.
    # Priority: _compiled (pre-built, batch path) > aliases (single-call convenience) > builtins.
    if _compiled is not None:
        c = _compiled
    elif aliases is not None:
        c = _compile_aliases(aliases)
    else:
        c = None

    if c is not None:
        label_names = c.label_names
        all_phrases_ = c.all_phrases
        alias_toksets_ = c.alias_toksets
        alias_toklists_ = c.alias_toklists
        excl_toksets_ = c.excl_toksets
    else:
        label_names = LABEL_NAMES
        all_phrases_ = _ALL_PHRASES
        alias_toksets_ = _ALIAS_TOKEN_SETS
        alias_toklists_ = _ALIAS_TOKEN_LISTS
        excl_toksets_ = _EXCLUDE_TOKEN_SETS

    # Pass 1: compute status + relational neighborhood for every Observation seed.
    entity_status: dict[str, str] = {}
    token_sets: dict[str, set[str]] = {}
    token_maps: dict[str, dict[str, list[int]]] = {}
    for eid, ent in entities.items():
        status = canonical_status(ent.get("label", ""), ent.get("relations", []))
        if status is None:
            continue
        entity_status[eid] = status
        ts, tm = _candidate_token_map(eid, entities)
        token_sets[eid] = ts
        token_maps[eid] = tm

    # Pass 2: per-disease alias matching.
    labels: dict[str, str] = {}
    matches: list[dict] = []

    for dz in label_names:
        excl_sets = excl_toksets_[dz] if apply_exclude else []

        # alias_phrase -> [(status, indices), ...]
        alias_hits: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
        for eid, ts in token_sets.items():
            tmap = token_maps[eid]
            seed_status = entity_status[eid]

            if excl_sets and any(ex.issubset(ts) for ex in excl_sets):
                continue

            for phrase, alias_tokset, alias_toklist in zip(
                all_phrases_[dz], alias_toksets_[dz], alias_toklists_[dz]
            ):
                if not alias_tokset.issubset(ts):
                    continue
                idx_list = _indices_for_alias(tmap, alias_toklist)
                if idx_list is None:
                    continue

                s = seed_status
                if s == "uncertain":
                    if uncertainty_policy == "as_positive":
                        s = "definitely present"
                    elif uncertainty_policy == "as_negative":
                        s = "definitely absent"
                    elif uncertainty_policy == "drop":
                        continue
                alias_hits[phrase].append((s, idx_list))

        if not alias_hits:
            continue

        phrase_statuses: list[str] = []
        for phrase, hits in alias_hits.items():
            final_status = _collapse_statuses(s for s, _ in hits)
            idx_list = next((ix for s, ix in hits if s == final_status), hits[0][1])
            matches.append({"disease": dz, "alias": phrase, "label": final_status, "start_ix": idx_list})
            phrase_statuses.append(final_status)

        labels[dz] = _collapse_statuses(phrase_statuses)

    return labels, matches, text
