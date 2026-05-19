"""High-level Python API and convenience exports for ``radlabels``.

Typical usage::

    from radlabels import label_reports

    results = label_reports([
        "FINDINGS: Small left pleural effusion. Cardiomegaly is stable.",
        "Bibasilar atelectasis without focal consolidation.",
    ])
    print(results[0].labels)
    for m in results[0].matches:
        print(m["disease"], m["alias"], m["label"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .aliases import ALIAS_VERSION, ALIASES, PARENT_MAP
from .matcher import LABEL_NAMES, _compile_aliases, label_study
from ._validation import validate_aliases

__all__ = [
    "ALIAS_VERSION",
    "ALIASES",
    "LABEL_NAMES",
    "PARENT_MAP",
    "label_study",
    "label_reports",
    "validate_aliases",
    "ReportResult",
]

__version__ = "0.1.0"


@dataclass
class ReportResult:
    """A labelled report: structured output of :func:`label_reports`."""

    report_id: str
    text: str
    labels: dict[str, str] = field(default_factory=dict)
    matches: list[dict] = field(default_factory=list)


def label_reports(
    texts: Sequence[str],
    *,
    ids: Sequence[str] | None = None,
    gpu: int | None = None,
    gpus: Sequence[int] | None = None,
    aliases: dict | None = None,
    apply_exclude: bool = True,
    uncertainty_policy: str = "keep",
) -> list[ReportResult]:
    """Run RadGraph + the alias matcher on a batch of reports.

    Parameters
    ----------
    texts
        Free-text radiology reports.
    ids
        Optional report identifiers. Defaults to ``["report_0001", ...]``.
    gpu, gpus
        See :func:`radlabels.radgraph_runner.run_radgraph`.
    aliases
        Custom alias dictionary following the ``ALIASES`` schema.  When
        provided, it **fully replaces** the built-in dictionary for every
        report.  Pass ``None`` (default) to use the built-in dictionary.
    apply_exclude, uncertainty_policy
        Forwarded to :func:`label_study`.
    """
    from .radgraph_runner import run_radgraph

    if ids is None:
        ids = [f"report_{i + 1:04d}" for i in range(len(texts))]
    if len(ids) != len(texts):
        raise ValueError("len(ids) must equal len(texts)")

    annotations = run_radgraph(list(texts), gpu=gpu, gpus=gpus)
    # Compile custom aliases once for the whole batch to avoid per-report overhead.
    compiled = _compile_aliases(aliases) if aliases is not None else None
    out: list[ReportResult] = []
    for rid, anno in zip(ids, annotations):
        labels, matches, parsed_text = label_study(
            anno,
            _compiled=compiled,
            apply_exclude=apply_exclude,
            uncertainty_policy=uncertainty_policy,
        )
        out.append(ReportResult(
            report_id=rid,
            text=parsed_text,
            labels=labels,
            matches=matches,
        ))
    return out
