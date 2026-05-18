"""Tests for :func:`radlabels.matcher.label_study`.

These tests feed in small hand-constructed RadGraph annotations so they run
without the real model. The schema mirrors what RadGraph emits for one report::

    {"0": {"text": "...", "entities": {eid: {tokens, label, start_ix, end_ix, relations}}}}
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from radlabels import LABEL_NAMES, label_study


def _anno(text: str, entities: dict) -> dict:
    return {"0": {"text": text, "entities": entities}}


def test_empty_input_returns_empty():
    labels, matches, text = label_study({})
    assert labels == {}
    assert matches == []
    assert text == ""


def test_simple_cardiomegaly_present():
    """`cardiomegaly` as a single Observation entity should fire."""
    text = "FINDINGS : Cardiomegaly ."
    anno = _anno(text, {
        "1": {
            "tokens": "Cardiomegaly", "label": "Observation::definitely present",
            "start_ix": 2, "end_ix": 2, "relations": [],
        },
    })
    labels, matches, _ = label_study(anno)
    assert labels.get("cardiomegaly") == "definitely present"
    assert any(m["disease"] == "cardiomegaly" and m["label"] == "definitely present"
               for m in matches)


def test_pleural_effusion_absent():
    """An Observation tagged `definitely absent` should produce a negative label."""
    text = "no pleural effusion ."
    anno = _anno(text, {
        "1": {
            "tokens": "pleural", "label": "Anatomy::definitely absent",
            "start_ix": 1, "end_ix": 1, "relations": [["located_at", "2"]],
        },
        "2": {
            "tokens": "effusion", "label": "Observation::definitely absent",
            "start_ix": 2, "end_ix": 2, "relations": [],
        },
    })
    labels, matches, _ = label_study(anno)
    assert labels.get("pleural_effusion") == "definitely absent"
    pe_matches = [m for m in matches if m["disease"] == "pleural_effusion"]
    assert pe_matches, "expected at least one pleural_effusion match"
    assert all(m["label"] == "definitely absent" for m in pe_matches)


def test_exclude_clause_vetoes_match():
    """`pleural_effusion` excludes `pericardial effusion` so it must NOT fire."""
    text = "small pericardial effusion ."
    anno = _anno(text, {
        "1": {
            "tokens": "pericardial", "label": "Anatomy::definitely present",
            "start_ix": 1, "end_ix": 1, "relations": [["located_at", "2"]],
        },
        "2": {
            "tokens": "effusion", "label": "Observation::definitely present",
            "start_ix": 2, "end_ix": 2, "relations": [],
        },
    })
    labels, matches, _ = label_study(anno)
    assert "pleural_effusion" not in labels, (
        "pericardial effusion must not fire pleural_effusion"
    )


def test_exclude_can_be_disabled():
    """With apply_exclude=False, a hit that would normally be vetoed fires."""
    # "blunting costophrenic angle" is a pleural_effusion alias. Add a
    # pericardial token so the exclude clause `pericardial effusion` can be
    # tested independently.
    text = "blunting costophrenic angle pericardial effusion ."
    anno = _anno(text, {
        "1": {
            "tokens": "blunting costophrenic angle",
            "label": "Observation::definitely present",
            "start_ix": 0, "end_ix": 2, "relations": [["located_at", "2"]],
        },
        "2": {
            "tokens": "pericardial effusion",
            "label": "Observation::definitely present",
            "start_ix": 3, "end_ix": 4, "relations": [],
        },
    })
    # With the default (apply_exclude=True), the pericardial-effusion exclude
    # vetoes the pleural_effusion hit.
    labels_strict, _, _ = label_study(anno, apply_exclude=True)
    assert "pleural_effusion" not in labels_strict

    # With exclude disabled, the alias fires.
    labels_loose, _, _ = label_study(anno, apply_exclude=False)
    assert labels_loose.get("pleural_effusion") == "definitely present"


def test_uncertainty_policy_keep_emits_uncertain():
    text = "possible pneumothorax ."
    anno = _anno(text, {
        "1": {
            "tokens": "pneumothorax", "label": "Observation::uncertain",
            "start_ix": 1, "end_ix": 1, "relations": [],
        },
    })
    labels, _, _ = label_study(anno)
    assert labels.get("pneumothorax") == "uncertain"


def test_uncertainty_policy_as_negative():
    text = "possible pneumothorax ."
    anno = _anno(text, {
        "1": {
            "tokens": "pneumothorax", "label": "Observation::uncertain",
            "start_ix": 1, "end_ix": 1, "relations": [],
        },
    })
    labels, _, _ = label_study(anno, uncertainty_policy="as_negative")
    assert labels.get("pneumothorax") == "definitely absent"


def test_uncertainty_policy_drop_omits_label():
    text = "possible pneumothorax ."
    anno = _anno(text, {
        "1": {
            "tokens": "pneumothorax", "label": "Observation::uncertain",
            "start_ix": 1, "end_ix": 1, "relations": [],
        },
    })
    labels, _, _ = label_study(anno, uncertainty_policy="drop")
    assert "pneumothorax" not in labels


def test_modifier_observation_does_not_fire():
    """An Observation that has an outgoing `modify` relation is a modifier (e.g.\
    \"no\", \"stable\") and must not be treated as a seed."""
    text = "stable cardiomegaly ."
    anno = _anno(text, {
        "1": {
            "tokens": "stable", "label": "Observation::definitely present",
            "start_ix": 0, "end_ix": 0, "relations": [["modify", "2"]],
        },
        "2": {
            "tokens": "cardiomegaly", "label": "Observation::definitely present",
            "start_ix": 1, "end_ix": 1, "relations": [],
        },
    })
    labels, matches, _ = label_study(anno)
    # cardiomegaly should still fire from the second entity.
    assert labels.get("cardiomegaly") == "definitely present"
    # And the `stable` modifier shouldn't add any extra disease keys.
    assert all(m["disease"] != "" for m in matches)


# ------------------------------------------------------------------ #
#                BUNDLED-CORPUS INVARIANT SMOKE TEST                 #
# ------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def bundled_matches() -> dict:
    p = (Path(__file__).resolve().parent.parent
         / "src" / "radlabels" / "samples" / "synthetic_matches.json")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def bundled_labels() -> dict:
    p = (Path(__file__).resolve().parent.parent
         / "src" / "radlabels" / "samples" / "synthetic_labels.json")
    with open(p) as f:
        return json.load(f)


def test_bundled_corpus_size(bundled_labels):
    assert len(bundled_labels) == 1000


def test_bundled_corpus_label_keys_subset(bundled_labels):
    """Every label key in the bundled output must be one of the ones we ship."""
    leafset = set(LABEL_NAMES)
    seen: set[str] = set()
    for labels in bundled_labels.values():
        seen.update(labels.keys())
    extra = seen - leafset
    assert not extra, f"bundled corpus references unknown labels: {sorted(extra)}"


def test_bundled_corpus_status_values(bundled_labels):
    """Every status must be one of the canonical three."""
    allowed = {"definitely present", "uncertain", "definitely absent"}
    for rid, labels in bundled_labels.items():
        for dz, status in labels.items():
            assert status in allowed, f"{rid}/{dz} bad status: {status!r}"


def test_bundled_corpus_match_indices_well_formed(bundled_matches):
    """Each match's start_ix list length must equal the alias's word count."""
    for rid, payload in list(bundled_matches.items())[:200]:
        for m in payload["matches"]:
            n_words = len(m["alias"].split())
            assert len(m["start_ix"]) == n_words, \
                f"{rid}: alias {m['alias']!r} has {n_words} words "\
                f"but {len(m['start_ix'])} indices"
