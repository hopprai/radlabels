"""Integration tests for the radlabels CLI.

These tests use Click's CliRunner to invoke the CLI without spawning a
subprocess, so they run without a GPU or the real RadGraph model.  The
`label` command's RadGraph step is bypassed via ``--radgraph-cache``.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import orjson
import pytest
from click.testing import CliRunner

from radlabels.cli import main

# Minimal well-formed RadGraph annotation for one report.
_ANNO = {
    "0": {
        "text": "FINDINGS : Cardiomegaly .",
        "entities": {
            "1": {
                "tokens": "Cardiomegaly",
                "label": "Observation::definitely present",
                "start_ix": 2,
                "end_ix": 2,
                "relations": [],
            }
        },
    }
}

_RID = "r001"


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def cache_file(tmp_path):
    """Write a single-report RadGraph cache JSON and return its path."""
    p = tmp_path / "cache.json"
    p.write_bytes(orjson.dumps({_RID: _ANNO}))
    return p


@pytest.fixture()
def reports_file(tmp_path):
    """Write a single-report input JSON and return its path."""
    p = tmp_path / "reports.json"
    p.write_bytes(orjson.dumps({_RID: "FINDINGS : Cardiomegaly ."}))
    return p


# ------------------------------------------------------------------ #
#                     basic cache-based label run                    #
# ------------------------------------------------------------------ #
def test_label_with_cache_exits_zero(runner, cache_file, reports_file):
    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_file),
        "--n-show", "0",
    ])
    assert result.exit_code == 0, result.output


def test_label_with_cache_produces_labels(runner, cache_file, reports_file, tmp_path):
    out = tmp_path / "labels.json"
    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_file),
        "--out", str(out),
        "--n-show", "0",
    ])
    assert result.exit_code == 0, result.output
    labels = orjson.loads(out.read_bytes())
    assert _RID in labels
    assert labels[_RID].get("cardiomegaly") == "definitely present"


# ------------------------------------------------------------------ #
#                 save-cache round-trip via tmp file                 #
# ------------------------------------------------------------------ #
def test_save_and_reload_cache(runner, reports_file, tmp_path):
    """Save cache from a first run, reload it in a second — same labels."""
    cache_path = tmp_path / "saved_cache.json"
    out1 = tmp_path / "labels1.json"
    out2 = tmp_path / "labels2.json"

    # Simulate a cache that was already produced (bypasses live inference).
    cache_path.write_bytes(orjson.dumps({_RID: _ANNO}))

    # First "run": load from the pre-written cache and write labels.
    r1 = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_path),
        "--out", str(out1),
        "--n-show", "0",
    ])
    assert r1.exit_code == 0, r1.output

    # Second run: reload the same cache, verify labels are identical.
    r2 = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_path),
        "--out", str(out2),
        "--n-show", "0",
    ])
    assert r2.exit_code == 0, r2.output
    assert orjson.loads(out1.read_bytes()) == orjson.loads(out2.read_bytes())


# ------------------------------------------------------------------ #
#                     --custom-aliases flag                          #
# ------------------------------------------------------------------ #
def test_custom_aliases_replaces_builtin_via_cli(runner, cache_file, reports_file, tmp_path):
    custom = {
        "my_finding": {"aliases": ["cardiomegaly"], "exclude": []}
    }
    aliases_file = tmp_path / "custom.json"
    aliases_file.write_bytes(orjson.dumps(custom))
    out = tmp_path / "labels.json"

    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_file),
        "--custom-aliases", str(aliases_file),
        "--out", str(out),
        "--n-show", "0",
    ])
    assert result.exit_code == 0, result.output
    labels = orjson.loads(out.read_bytes())
    assert "my_finding" in labels[_RID]
    assert "cardiomegaly" not in labels[_RID]


# ------------------------------------------------------------------ #
#                         error cases                                #
# ------------------------------------------------------------------ #
def test_missing_cache_id_exits_nonzero(runner, reports_file, tmp_path):
    """A cache that doesn't contain the requested report ID must fail."""
    cache = tmp_path / "cache.json"
    cache.write_bytes(orjson.dumps({"wrong_id": _ANNO}))

    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache),
        "--n-show", "0",
    ])
    assert result.exit_code != 0


def test_malformed_custom_aliases_exits_nonzero(runner, cache_file, reports_file, tmp_path):
    """A custom-aliases file with schema errors must abort."""
    bad = {"bad_label": "not a dict"}
    aliases_file = tmp_path / "bad_aliases.json"
    aliases_file.write_bytes(orjson.dumps(bad))

    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache_file),
        "--custom-aliases", str(aliases_file),
        "--n-show", "0",
    ])
    assert result.exit_code != 0


def test_mutually_exclusive_cache_flags(runner, reports_file, tmp_path):
    """--radgraph-cache and --save-radgraph-cache must not be used together."""
    cache = tmp_path / "cache.json"
    cache.write_bytes(orjson.dumps({_RID: _ANNO}))
    save = tmp_path / "out_cache.json"

    result = runner.invoke(main, [
        "label",
        "--file", str(reports_file),
        "--radgraph-cache", str(cache),
        "--save-radgraph-cache", str(save),
        "--n-show", "0",
    ])
    assert result.exit_code != 0
