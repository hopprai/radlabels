"""Label the bundled corpus of 1000 reports and print a summary table.

The bundled labels are pre-computed; this example uses them directly so it
runs in milliseconds with no GPU. Use ``radlabels demo --recompute`` to run
RadGraph on the bundled reports instead.
"""
from pathlib import Path

import orjson
from radlabels.formatting import corpus_summary_table, make_console

samples = Path(__file__).resolve().parent.parent / "src" / "radlabels" / "samples"

with open(samples / "synthetic_labels.json", "rb") as f:
    labels_by_id = orjson.loads(f.read())

console = make_console()
console.print(corpus_summary_table(list(labels_by_id.values())))
