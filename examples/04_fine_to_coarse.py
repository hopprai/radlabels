"""Derive parent (coarse) labels from leaf (fine) label_study output.

The paper trains image models with both fine leaves and their parent groups
(Appendix C, fine-to-coarse aggregation).  This example shows how to
reproduce that mapping using ``PARENT_MAP``.

Two aggregation modes are shown:

1. **Rule-based** (report labels): a parent is ``definitely present`` if any
   child is; ``uncertain`` if any child is uncertain and none are present;
   ``definitely absent`` if all children are absent.

2. **Model-based** (numeric scores): a parent score is the max of its
   children's scores, following Appendix C of the paper.

Usage::

    python examples/04_fine_to_coarse.py
"""
from __future__ import annotations

from collections import defaultdict

from radlabels import PARENT_MAP


# ------------------------------------------------------------------ #
#                   rule-based aggregation                           #
# ------------------------------------------------------------------ #
_PRIORITY = ("definitely present", "uncertain", "definitely absent")


def fine_to_coarse_labels(
    fine_labels: dict[str, str],
) -> dict[str, str]:
    """Aggregate fine leaf labels to their parent groups.

    Parameters
    ----------
    fine_labels
        ``{leaf_label: status}`` dict as returned by ``label_study``.
        Missing keys are treated as "not discussed" (not included).

    Returns
    -------
    dict[str, str]
        Parent labels that have at least one child with a status.
        Keys are the 10 parent group names; values are the collapsed status.
    """
    parent_statuses: dict[str, list[str]] = defaultdict(list)
    for leaf, status in fine_labels.items():
        parent = PARENT_MAP.get(leaf)
        if parent is not None:
            parent_statuses[parent].append(status)

    result: dict[str, str] = {}
    for parent, statuses in parent_statuses.items():
        s_set = set(statuses)
        for priority in _PRIORITY:
            if priority in s_set:
                result[parent] = priority
                break
    return result


# ------------------------------------------------------------------ #
#                  model-based aggregation                           #
# ------------------------------------------------------------------ #
def fine_to_coarse_scores(
    fine_scores: dict[str, float],
) -> dict[str, float]:
    """Aggregate fine leaf model scores to parent groups via max pooling.

    Parameters
    ----------
    fine_scores
        ``{leaf_label: score}`` dict where scores are floats (e.g., predicted
        probabilities from a multi-label classifier).

    Returns
    -------
    dict[str, float]
        Parent scores: max over all children that appear in ``fine_scores``.
    """
    parent_scores: dict[str, list[float]] = defaultdict(list)
    for leaf, score in fine_scores.items():
        parent = PARENT_MAP.get(leaf)
        if parent is not None:
            parent_scores[parent].append(score)
    return {parent: max(scores) for parent, scores in parent_scores.items()}


# ------------------------------------------------------------------ #
#                           demo                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Simulate label_study output for one report.
    example_fine_labels = {
        "pleural_effusion": "definitely present",
        "pleural_thickening": "definitely absent",
        "atelectasis": "uncertain",
        "cardiomegaly": "definitely present",
        "pacemaker_electronic_cardiac_device_or_wires": "definitely present",
    }

    coarse = fine_to_coarse_labels(example_fine_labels)

    print("Fine labels:")
    for k, v in sorted(example_fine_labels.items()):
        parent = PARENT_MAP.get(k, "—")
        print(f"  {k:45s} {v:20s} → {parent}")

    print("\nCoarse (parent) labels:")
    for k, v in sorted(coarse.items()):
        print(f"  {k:45s} {v}")

    # Simulate model probability scores.
    example_fine_scores = {
        "pleural_effusion": 0.92,
        "pleural_thickening": 0.15,
        "atelectasis": 0.61,
        "cardiomegaly": 0.87,
        "pacemaker_electronic_cardiac_device_or_wires": 0.98,
    }

    coarse_scores = fine_to_coarse_scores(example_fine_scores)
    print("\nCoarse (parent) max-pooled scores:")
    for k, v in sorted(coarse_scores.items()):
        print(f"  {k:45s} {v:.3f}")
