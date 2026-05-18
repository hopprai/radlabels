"""Rich-based pretty printers for matcher output.

Two helpers:

- :func:`per_report_table` \u2014 one row per matched alias for a single report.
- :func:`corpus_summary_table` \u2014 per-label present/uncertain/absent counts
  across an entire batch of reports.

Both return :class:`rich.table.Table` instances so the caller can render
them via ``rich.console.Console.print``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Color scheme aligned with status semantics.
_STATUS_STYLE = {
    "definitely present": "bold green",
    "uncertain": "yellow",
    "definitely absent": "dim red",
}


def _color_status(status: str) -> Text:
    return Text(status, style=_STATUS_STYLE.get(status, ""))


def per_report_table(
    *,
    report_id: str,
    text: str,
    labels: dict[str, str],
    matches: list[dict],
    show_text: bool = True,
) -> list:
    """Return rich renderables for one labelled report.

    The output is a list of objects (a panel with the report text, then a
    table of alias matches). Pass them to :class:`rich.console.Console.print`.
    """
    out: list = []
    if show_text and text:
        out.append(Panel(text, title=f"[bold]{report_id}[/]", expand=False, border_style="cyan"))

    if not matches:
        out.append(Text(f"  ({report_id}) no alias matches", style="dim"))
        return out

    table = Table(
        title=f"Matches \u2014 {report_id}  ({len(matches)} hits, "
              f"{len(labels)} distinct labels)",
        title_justify="left",
        show_lines=False,
        header_style="bold magenta",
    )
    table.add_column("Disease", style="cyan", no_wrap=True)
    table.add_column("Alias", style="white")
    table.add_column("Status", no_wrap=True)
    table.add_column("Token positions", justify="right", style="dim")

    sorted_matches = sorted(
        matches, key=lambda m: (m["disease"], m["alias"])
    )
    for m in sorted_matches:
        table.add_row(
            m["disease"],
            m["alias"],
            _color_status(m["label"]),
            ", ".join(str(i) for i in m["start_ix"]),
        )
    out.append(table)
    return out


def corpus_summary_table(
    all_labels: Sequence[dict[str, str]],
    *,
    label_names: Iterable[str] | None = None,
    title: str = "Corpus summary",
) -> Table:
    """Return a per-label present/uncertain/absent counts table.

    ``all_labels`` is a list of per-report ``label_study`` outputs.

    Labels that have at least one non-zero entry are sorted by total
    occurrences (descending) so the most frequently fired labels come first.
    Labels with zero hits across the whole corpus are omitted.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for labels in all_labels:
        for dz, status in labels.items():
            counts[dz][status] += 1

    if label_names is None:
        label_names = list(counts.keys())

    rows = []
    for dz in label_names:
        c = counts.get(dz, Counter())
        prs = c["definitely present"]
        unc = c["uncertain"]
        absn = c["definitely absent"]
        total = prs + unc + absn
        if total == 0:
            continue
        rows.append((dz, prs, unc, absn, total))
    rows.sort(key=lambda r: (-r[4], r[0]))

    table = Table(
        title=f"{title}  ({len(all_labels)} reports, {len(rows)} labels with hits)",
        title_justify="left",
        show_lines=False,
        header_style="bold magenta",
    )
    table.add_column("Disease", style="cyan", no_wrap=True)
    table.add_column("Present", justify="right", style="green")
    table.add_column("Uncertain", justify="right", style="yellow")
    table.add_column("Absent", justify="right", style="red")
    table.add_column("Total", justify="right", style="bold")

    for dz, prs, unc, absn, total in rows:
        table.add_row(dz, str(prs), str(unc), str(absn), str(total))

    return table


def make_console() -> Console:
    """Return a stdout console with sensible defaults.

    Auto-sizes to the terminal when stdout is a TTY. Honors the
    ``RADLABELS_WIDTH`` env var to override (e.g. for non-TTY captured
    output).
    """
    import os
    import sys
    if sys.stdout.isatty() and "RADLABELS_WIDTH" not in os.environ:
        return Console()
    width = int(os.environ.get("RADLABELS_WIDTH", "140"))
    return Console(width=width, force_terminal=True)
