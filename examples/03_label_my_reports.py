"""Label a JSON file of your own reports and write the labels to disk.

Input  format (``my_reports.json``)::

    {"r0001": "FINDINGS: ...", "r0002": "FINDINGS: ...", ...}

Output format (``my_labels.json``)::

    {"r0001": {"pleural_effusion": "definitely present", ...}, ...}
"""
import sys
from pathlib import Path

import orjson
from radlabels import label_reports


def main(in_path: str, out_path: str) -> None:
    with open(in_path, "rb") as f:
        data = orjson.loads(f.read())
    rids, texts = list(data.keys()), list(data.values())
    results = label_reports(texts, ids=rids)

    out = {r.report_id: r.labels for r in results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(out, option=orjson.OPT_INDENT_2))
    print(f"Wrote {len(out)} labelled reports to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
