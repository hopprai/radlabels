"""Command-line interface for radlabels."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import click
import orjson

from . import label_study  # noqa: F401
from .aliases import ALIASES, ALIAS_VERSION  # noqa: F401
from .formatting import (
    corpus_summary_table,
    make_console,
    per_report_table,
)

log = logging.getLogger("radlabels")


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
@click.option("--radgraph-cache", "rg_cache_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="Load pre-computed RadGraph annotations from this JSON file "
                   '({report_id: annotation}) and skip inference.')
@click.option("--save-radgraph-cache", "save_rg_cache_path",
              type=click.Path(dir_okay=False), default=None,
              help="After running RadGraph inference, save raw annotations to this path.")
@click.option("--custom-aliases", "custom_aliases_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON file with a custom alias dictionary (fully replaces built-ins).")
@click.option("--verbose", is_flag=True, default=False,
              help="Log alias version and run metadata to stderr.")
def label(
    inline_text: str | None,
    in_file: str | None,
    out_path: str | None,
    matches_path: str | None,
    n_show: int,
    gpu: int | None,
    gpus: str | None,
    rg_cache_path: str | None,
    save_rg_cache_path: str | None,
    custom_aliases_path: str | None,
    verbose: bool,
) -> None:
    """Label one or many reports."""
    if inline_text is None and in_file is None:
        raise click.UsageError("Pass either --text or --file.")
    if inline_text is not None and in_file is not None:
        raise click.UsageError("Pass --text OR --file, not both.")
    if rg_cache_path and save_rg_cache_path:
        raise click.UsageError(
            "--radgraph-cache and --save-radgraph-cache are mutually exclusive."
        )

    # ---- verbose logging setup ------------------------------------------
    if verbose:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                            format="%(message)s")
        if custom_aliases_path:
            log.info("radlabels v%s  alias_version=custom  custom_aliases=%s",
                     _radlabels_version(), custom_aliases_path)
        else:
            log.info("radlabels v%s  alias_version=%s",
                     _radlabels_version(), ALIAS_VERSION)

    # ---- resolve alias dictionary ---------------------------------------
    aliases = _load_custom_aliases(custom_aliases_path) if custom_aliases_path else None

    # ---- parse input IDs + texts ----------------------------------------
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

    # ---- run RadGraph or load from cache --------------------------------
    if rg_cache_path:
        annotations = _load_rg_cache(rg_cache_path, rids)
    else:
        gpu_list: Sequence[int] | None = None
        if gpus:
            gpu_list = [int(x) for x in gpus.split(",") if x.strip()]

        from .radgraph_runner import run_radgraph
        annotations = run_radgraph(texts, gpu=gpu, gpus=gpu_list)

        if save_rg_cache_path:
            cache_dump = {rid: anno for rid, anno in zip(rids, annotations)}
            Path(save_rg_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_rg_cache_path, "wb") as f:
                f.write(orjson.dumps(cache_dump, option=orjson.OPT_INDENT_2))

    # ---- run matcher ----------------------------------------------------
    results = []
    for rid, anno in zip(rids, annotations):
        lbl, matches_list, text = label_study(
            anno,
            aliases=aliases,
        )
        from . import ReportResult
        results.append(ReportResult(
            report_id=rid,
            text=text,
            labels=lbl,
            matches=matches_list,
        ))

    # ---- write outputs --------------------------------------------------
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


def _radlabels_version() -> str:
    from . import __version__
    return __version__


def _load_rg_cache(cache_path: str, rids: list[str]) -> list[dict]:
    """Load pre-computed RadGraph annotations and validate them."""
    with open(cache_path, "rb") as f:
        cache: dict = orjson.loads(f.read())

    # Check all requested IDs are present.
    missing = [rid for rid in rids if rid not in cache]
    if missing:
        preview = missing[:10]
        extra = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise click.ClickException(
            f"--radgraph-cache is missing {len(missing)} report ID(s): "
            + ", ".join(preview) + extra
        )

    # Validate each entry has the expected structure.
    malformed = []
    for rid in rids:
        anno = cache[rid]
        if not isinstance(anno, dict):
            malformed.append(rid)
            continue
        inner = anno.get("0")
        if not isinstance(inner, dict) or "entities" not in inner:
            malformed.append(rid)
    if malformed:
        preview = malformed[:5]
        extra = f" (and {len(malformed) - 5} more)" if len(malformed) > 5 else ""
        raise click.ClickException(
            f"--radgraph-cache has malformed entries for: "
            + ", ".join(preview) + extra
            + '. Expected {"0": {"text": ..., "entities": {...}}}.'
        )

    return [cache[rid] for rid in rids]


def _load_custom_aliases(path: str) -> dict:
    """Load and validate a custom alias JSON file."""
    from ._validation import validate_aliases

    with open(path, "rb") as f:
        aliases = orjson.loads(f.read())

    if not isinstance(aliases, dict):
        raise click.ClickException(
            f"--custom-aliases: {path} must be a JSON object."
        )

    messages = validate_aliases(aliases)
    errors = [m for m in messages if m.startswith("ERROR")]
    warnings = [m for m in messages if m.startswith("WARNING")]

    if errors:
        for e in errors:
            click.echo(e, err=True)
        raise click.ClickException(
            f"--custom-aliases: {len(errors)} schema error(s) in {path}. Aborting."
        )
    for w in warnings:
        click.echo(w, err=True)

    return aliases


if __name__ == "__main__":
    main()
