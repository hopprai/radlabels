"""Command-line interface for radlabels."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import click
import orjson

from . import label_study
from .aliases import ALIASES
from .formatting import (
    corpus_summary_table,
    make_console,
    per_report_table,
)


@click.group()
@click.version_option()
def main() -> None:
    """Turn radiology report text into disease labels."""


# ------------------------------------------------------------------ #
#                              demo                                  #
# ------------------------------------------------------------------ #
@main.command()
@click.option("--n", "n_show", default=5, show_default=True,
              help="Show per-report tables for the first N reports.")
@click.option("--recompute/--cached", default=False,
              help="Re-run RadGraph + matcher instead of loading cached labels "
                   "(requires GPU recommended).")
@click.option("--show-text/--no-text", default=True,
              help="Print each shown report's text alongside its match table.")
def demo(n_show: int, recompute: bool, show_text: bool) -> None:
    """Run on the bundled set of 1000 reports and print a formatted table."""
    console = make_console()
    samples = _samples_dir()
    reports_dir = samples / "synthetic_reports"
    if not reports_dir.exists():
        console.print(
            f"[red]No reports directory at {reports_dir}.[/]"
            " Did the package install correctly?"
        )
        sys.exit(2)

    report_paths = sorted(reports_dir.glob("*.txt"))
    if not report_paths:
        console.print(f"[red]No .txt reports found in {reports_dir}.[/]")
        sys.exit(2)
    rids = [p.stem for p in report_paths]
    texts = [p.read_text() for p in report_paths]

    if recompute:
        console.print(
            f"[bold]Re-running RadGraph on {len(texts)} reports[/] (this can take a "
            "while; uses GPU if available)..."
        )
        from . import label_reports as _label_reports

        results = _label_reports(texts, ids=rids)
        all_labels = [r.labels for r in results]
        items = [(r.report_id, r.text, r.labels, r.matches) for r in results]
    else:
        labels_path = samples / "synthetic_labels.json"
        matches_path = samples / "synthetic_matches.json"
        if not labels_path.exists() or not matches_path.exists():
            console.print(
                f"[red]Cached labels not found at {labels_path}. Try "
                "`radlabels demo --recompute`.[/]"
            )
            sys.exit(2)

        with open(labels_path, "rb") as f:
            cached_labels = orjson.loads(f.read())
        with open(matches_path, "rb") as f:
            cached_matches = orjson.loads(f.read())

        all_labels = [cached_labels.get(rid, {}) for rid in rids]
        items = [
            (rid, text, cached_labels.get(rid, {}),
             cached_matches.get(rid, {}).get("matches", []))
            for rid, text in zip(rids, texts)
        ]

    # Per-report tables for the first N.
    for rid, text, labels, matches in items[: max(n_show, 0)]:
        console.print()
        for r in per_report_table(
            report_id=rid, text=text, labels=labels, matches=matches,
            show_text=show_text,
        ):
            console.print(r)

    # Corpus-level summary across ALL reports.
    console.print()
    console.print(corpus_summary_table(all_labels, title="Bundled corpus summary"))


# ------------------------------------------------------------------ #
#                              label                                 #
# ------------------------------------------------------------------ #
@main.command()
@click.option("--text", "inline_text", default=None,
              help="A single report passed inline.")
@click.option("--file", "in_file", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help='JSON file mapping report IDs to text: {"r1": "FINDINGS: ...", ...}.')
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Write per-report labels JSON to this path.")
@click.option("--matches", "matches_path", type=click.Path(dir_okay=False), default=None,
              help="Write per-report alias matches JSON to this path.")
@click.option("--n-show", default=3, show_default=True,
              help="Number of per-report tables to print to stdout.")
@click.option("--gpu", type=int, default=None,
              help="GPU index (single device). Defaults to autodetect.")
@click.option("--gpus", default=None,
              help='Comma-separated GPU indices for data-parallel inference, e.g. "0,1,2".')
def label(
    inline_text: str | None,
    in_file: str | None,
    out_path: str | None,
    matches_path: str | None,
    n_show: int,
    gpu: int | None,
    gpus: str | None,
) -> None:
    """Label one or many reports."""
    if inline_text is None and in_file is None:
        raise click.UsageError("Pass either --text or --file.")
    if inline_text is not None and in_file is not None:
        raise click.UsageError("Pass --text OR --file, not both.")

    if inline_text is not None:
        rids = ["report_0001"]
        texts = [inline_text]
    else:
        with open(in_file, "rb") as f:
            data = orjson.loads(f.read())
        if not isinstance(data, dict) or not data:
            raise click.UsageError(
                f"{in_file} must be a non-empty JSON object mapping ids to text."
            )
        rids = list(data.keys())
        texts = [data[k] if isinstance(data[k], str) else "" for k in rids]

    gpu_list: Sequence[int] | None = None
    if gpus:
        gpu_list = [int(x) for x in gpus.split(",") if x.strip()]

    from . import label_reports as _label_reports

    results = _label_reports(texts, ids=rids, gpu=gpu, gpus=gpu_list)

    if out_path:
        labels_dump = {r.report_id: r.labels for r in results}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(labels_dump, option=orjson.OPT_INDENT_2))
    if matches_path:
        matches_dump = {r.report_id: {"text": r.text, "matches": r.matches}
                        for r in results}
        Path(matches_path).parent.mkdir(parents=True, exist_ok=True)
        with open(matches_path, "wb") as f:
            f.write(orjson.dumps(matches_dump, option=orjson.OPT_INDENT_2))

    console = make_console()
    for r in results[: max(n_show, 0)]:
        console.print()
        for renderable in per_report_table(
            report_id=r.report_id, text=r.text, labels=r.labels, matches=r.matches,
        ):
            console.print(renderable)

    if len(results) > 1:
        console.print()
        console.print(corpus_summary_table([r.labels for r in results]))


# ------------------------------------------------------------------ #
#                          internal helpers                          #
# ------------------------------------------------------------------ #
def _samples_dir() -> Path:
    """Return the absolute path to the bundled samples directory."""
    return Path(__file__).resolve().parent / "samples"


# Touch the import so static analyzers don't think `label_study` is unused
# in this module; it's exposed via ``from radlabels import label_study``.
_ = label_study
_ = ALIASES


if __name__ == "__main__":
    main()
