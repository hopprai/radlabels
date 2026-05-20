"""Smoke tests for the examples/ scripts.

Examples that call label_reports() (which requires RadGraph + GPU) are not
imported here.  We test the pure-Python logic in examples 02 and 04 directly,
and verify that all example files are importable without error.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _import_example(name: str):
    """Import an example module by filename without executing __main__ block."""
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ #
#                     importability (all examples)                   #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("filename", [
    "02_run_demo_set.py",
    "04_fine_to_coarse.py",
])
def test_example_importable(filename):
    """Each example must import without raising."""
    _import_example(filename)


# ------------------------------------------------------------------ #
#                 02_run_demo_set — corpus summary                   #
# ------------------------------------------------------------------ #
def test_02_corpus_summary_runs():
    """The bundled 1000-report summary must complete without error."""
    from pathlib import Path as _Path
    import orjson
    from radlabels.formatting import corpus_summary_table, make_console

    samples = _Path(__file__).resolve().parent.parent / "src" / "radlabels" / "samples"
    with open(samples / "synthetic_labels.json", "rb") as f:
        labels_by_id = orjson.loads(f.read())

    table = corpus_summary_table(list(labels_by_id.values()))
    assert table is not None


# ------------------------------------------------------------------ #
#                 04_fine_to_coarse — aggregation logic              #
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def ftc():
    return _import_example("04_fine_to_coarse.py")


def test_04_fine_to_coarse_labels_present(ftc):
    fine = {"pleural_effusion": "definitely present", "pleural_thickening": "definitely absent"}
    coarse = ftc.fine_to_coarse_labels(fine)
    assert coarse["pleural_abnormality"] == "definitely present"


def test_04_fine_to_coarse_labels_priority(ftc):
    """present beats uncertain beats absent."""
    fine = {
        "atelectasis": "uncertain",
        "lung_nodule_or_mass": "definitely present",
        "infiltration": "definitely absent",
    }
    coarse = ftc.fine_to_coarse_labels(fine)
    assert coarse["pulmonary_abnormality"] == "definitely present"


def test_04_fine_to_coarse_labels_absent_only(ftc):
    fine = {"pneumothorax": "definitely absent"}
    coarse = ftc.fine_to_coarse_labels(fine)
    assert coarse["pleural_abnormality"] == "definitely absent"


def test_04_fine_to_coarse_scores_max_pool(ftc):
    fine = {"pleural_effusion": 0.9, "pleural_thickening": 0.3}
    coarse = ftc.fine_to_coarse_scores(fine)
    assert abs(coarse["pleural_abnormality"] - 0.9) < 1e-9


def test_04_fine_to_coarse_scores_unknown_leaf_ignored(ftc):
    """Leaves absent from PARENT_MAP must not raise."""
    fine = {"not_a_real_label": 0.5}
    coarse = ftc.fine_to_coarse_scores(fine)
    assert coarse == {}
