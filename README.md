# radlabels

**Convert radiology report text into structured disease labels.**

A small, radiologist-defined dictionary on top of [RadGraph-XL](https://github.com/Stanford-AIMI/radgraph).
No model training, no API keys, no fixed taxonomy: to add a new finding
you simply write a few phrases.

```bash
pip install radlabels
radlabels demo
```

The package ships with **49 findings**, **649 alias phrases**, and a
**1000-report demo corpus** with pre-computed labels so the demo runs
instantly without GPU.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Label your own reports](#1-label-your-own-reports)
3. [Run the bundled corpus](#2-run-the-bundled-corpus)
4. [Hardware and benchmarks](#hardware-and-benchmarks)
5. [Label set](#label-set)
6. [Tuning knobs](#tuning-knobs)
7. [Citation](#citation)
8. [License](#license)

---

## How it works

1. A radiologist defines a finding by listing a handful of phrases that name
   it, for example `pleural effusion`, `hydrothorax`, `fluid in the pleural
   space`.
2. RadGraph runs once over the report and extracts the clinical observations
   the radiologist actually wrote, each with a presence status
   (present / uncertain / absent).
3. The radiologist's phrases are matched against the parsed report. A study
   gets a finding if at least one phrase fits, with the same status the
   report gave it.
4. Every label points back to the exact tokens that fired it, so a reviewer
   can audit any label in seconds.

Editing or adding a finding is a phrase edit. The expensive RadGraph step runs
once per report and is never repeated.

---

## 1. Label your own reports

### Inline (single report)

```bash
radlabels label --text "FINDINGS: Small left pleural effusion. Cardiomegaly is stable."
```

### Batch (JSON file)

`my_reports.json`:

```json
{
  "r0001": "FINDINGS: Small left pleural effusion. Cardiomegaly is stable.",
  "r0002": "FINDINGS: Bibasilar atelectasis. No focal consolidation."
}
```

```bash
radlabels label --file my_reports.json --out labels.json --matches matches.json
```

### Output

`labels.json` (one entry per report, only fired findings):

```json
{
  "r0001": {
    "cardiomegaly": "definitely present",
    "pleural_effusion": "definitely present"
  },
  "r0002": {
    "atelectasis": "definitely present",
    "consolidation": "definitely absent"
  }
}
```

`matches.json` (per-alias hits with token positions, for audit / review):

```json
{
  "r0001": {
    "text": "FINDINGS : Small left pleural effusion . Cardiomegaly is stable .",
    "matches": [
      {"disease": "cardiomegaly",     "alias": "cardiomegaly",      "label": "definitely present", "start_ix": [6]},
      {"disease": "pleural_effusion", "alias": "pleural effusion",  "label": "definitely present", "start_ix": [3, 4]},
      {"disease": "pleural_effusion", "alias": "small effusion",    "label": "definitely present", "start_ix": [2, 4]}
    ]
  }
}
```

`start_ix` indices are positions in the whitespace-tokenized text from
RadGraph. Status values are always one of
`"definitely present"`, `"uncertain"`, `"definitely absent"`. **Missing
keys mean "no evidence found"** — do not treat them as definitely absent.

### Python API

```python
from radlabels import label_reports

results = label_reports([
    "FINDINGS: Small left pleural effusion. Cardiomegaly is stable.",
    "Bibasilar atelectasis without focal consolidation.",
])

for r in results:
    print(r.report_id, r.labels)
    for m in r.matches:
        print(" ", m["disease"], m["alias"], m["label"], m["start_ix"])
```

---

## 2. Run the bundled corpus

The package ships with 1000 chest radiology reports and pre-computed labels in
`samples/synthetic_reports/`, `samples/synthetic_labels.json`, and
`samples/synthetic_matches.json`. They are intended for demoing the pipeline
and are not for clinical use.

```bash
radlabels demo               # first 5 reports + corpus summary
radlabels demo --n 20        # show 20 per-report match tables
radlabels demo --recompute   # re-run RadGraph instead of using cached labels
```

### Example: a single report's match table

For `synth_0015`:

```text
FINDINGS:

The bilateral parenchymal opacities are slightly improved but continue to be
present right greater than left lower lobe greater than upper lobe.
Right-sided Port-A-Cath is unchanged. The NG tube is again seen in the neo
esophagus. ETT ends 5.5 cm above the carina. Right chest tube is unchanged.
There small bilateral pleural effusion. The ET tube is
```

`labels`:

| Disease             | Status             |
|---------------------|--------------------|
| `air_space_opacity` | definitely present |
| `enteric_tube`      | definitely present |
| `intercostal_drain` | definitely present |
| `lung_opacity`      | definitely present |
| `pleural_effusion`  | definitely present |

`matches`:

| Disease             | Alias                 | Status             | Tokens   |
|---------------------|-----------------------|--------------------|----------|
| `air_space_opacity` | `parenchymal opacity` | definitely present | `[4, 5]` |
| `enteric_tube`      | `ng tube`             | definitely present | `[37, 38]` |
| `intercostal_drain` | `chest tube`          | definitely present | `[58, 59]` |
| `lung_opacity`      | `parenchymal opacity` | definitely present | `[4, 5]` |
| `pleural_effusion`  | `pleural effusion`    | definitely present | `[66, 67]` |
| `pleural_effusion`  | `small effusion`      | definitely present | `[64, 67]` |

### Example: corpus-wide summary across all 1000 reports

| Disease                                              | Present | Uncertain | Absent | Total |
|------------------------------------------------------|--------:|----------:|-------:|------:|
| `pleural_effusion`                                   |     350 |        42 |     43 |   435 |
| `pneumothorax`                                       |      62 |        12 |    325 |   399 |
| `pulmonary_congestion_pulmonary_venous_congestion`   |     346 |         8 |     24 |   378 |
| `air_space_opacity`                                  |     353 |        11 |      6 |   370 |
| `cardiomegaly`                                       |     313 |         1 |      3 |   317 |
| `pulmonary_edema`                                    |     274 |        11 |     21 |   306 |
| `atelectasis`                                        |     286 |        10 |      0 |   296 |
| `pneumonia`                                          |     257 |        26 |      1 |   284 |
| `enteric_tube`                                       |     267 |         1 |      3 |   271 |
| `endotracheal_tube`                                  |     242 |         0 |      4 |   246 |
| `lung_opacity`                                       |     226 |         2 |      0 |   228 |
| `consolidation`                                      |     187 |         8 |      4 |   199 |
| `central_venous_catheter`                            |     188 |         0 |      0 |   188 |
| `intercostal_drain`                                  |     116 |         1 |      2 |   119 |
| `emphysema`                                          |     103 |         1 |      0 |   104 |
| `fracture_generic`                                   |      84 |         1 |     14 |    99 |
| `pacemaker_electronic_cardiac_device_or_wires`       |      90 |         0 |      0 |    90 |
| `calcification_of_the_aorta`                         |      76 |         0 |      0 |    76 |
| `tortuous_aorta`                                     |      74 |         0 |      0 |    74 |
| `subcutaneous_emphysema`                             |      72 |         0 |      0 |    72 |
| ...                                                  |     ... |       ... |    ... |   ... |

48 / 49 labels in the dictionary fire at least once on the bundled corpus.

---

## Hardware and benchmarks

`label_reports` autodetects CUDA. Pass `gpu=N` for a specific device or
`gpus=[0, 1, 2]` for data-parallel inference across multiple GPUs.

Reference numbers measured on this hardware:

- **GPU**: NVIDIA A100-SXM4-80GB
- **CPU**: AMD EPYC 7J13 (64-core) × 2 (128 cores total)
- **PyTorch**: 2.4.1+cu121, **transformers**: 4.x, **RadGraph**: 0.1.18

| Backend       | Reports | Wall time | Throughput  |
|---------------|--------:|----------:|------------:|
| 1 × A100 GPU  |     100 |   ~10.6 s |  9.5 reports/s |
| CPU only      |      30 |    ~3.3 s |  9.2 reports/s |

(Steady-state, model warmup excluded.)

Multi-GPU scales linearly; on a 3 × A100 box, labeling 100 k reports takes
about **22 minutes**.

---

## Label set

The shipped dictionary has **49 findings**. Add or rename any of them by
editing `radlabels/aliases.py` (or pass your own dict to the API).

| Label | # aliases | Example aliases |
|---|---|---|
| `acute_rib_fracture` | 8 | `acute rib fracture`, `new rib fracture`, `recent rib fracture` |
| `air_space_opacity` | 32 | `consolidation`, `infiltrate`, `pneumonia` |
| `atelectasis` | 21 | `atelectasis`, `atelectatic`, `atelectatic lung` |
| `bronchial_wall_thickening` | 7 | `bronchial wall thickening`, `bronchial thickening`, `airway wall thickening` |
| `bullous_disease` | 13 | `bullous changes`, `bullous change`, `pulmonary bullae` |
| `calcification_of_the_aorta` | 6 | `aortic calcification`, `calcified aorta`, `atherosclerotic calcification` |
| `cardiomegaly` | 18 | `cardiomegaly`, `enlarged heart`, `heart enlarged` |
| `central_venous_catheter` | 17 | `central venous catheter`, `central line`, `cvc` |
| `consolidation` | 7 | `consolidation`, `focal consolidation`, `lobar consolidation` |
| `emphysema` | 6 | `emphysema`, `centrilobular emphysema`, `paraseptal emphysema` |
| `endotracheal_tube` | 6 | `endotracheal tube`, `et tube`, `ett` |
| `enlarged_cardiomediastinum` | 7 | `enlarged cardiomediastinum`, `enlarged cardiomediastinal silhouette`, `widened mediastinum` |
| `enteric_tube` | 8 | `enteric tube`, `nasogastric tube`, `ng tube` |
| `fracture_generic` | 11 | `fracture`, `fractures`, `osseous fracture` |
| `ground_glass_opacity` | 12 | `ground glass opacity`, `ground-glass opacity`, `groundglass opacity` |
| `hiatus_hernia` | 10 | `hiatal hernia`, `hiatus hernia`, `paraesophageal hernia` |
| `hilar_lymphadenopathy` | 14 | `hilar adenopathy`, `hilar lymphadenopathy`, `enlarged hilar nodes` |
| `hyperinflation` | 20 | `hyperexpanded`, `hyperexpansion`, `hyperinflated` |
| `implantable_electronic_device` | 14 | `spinal cord stimulator`, `neurostimulator`, `nerve stimulator` |
| `infiltration` | 7 | `infiltrate`, `infiltrates`, `pulmonary infiltrate` |
| `intercostal_drain` | 16 | `pleural drain`, `pleural tube`, `chest tube` |
| `interstitial_thickening` | 19 | `interstitial markings`, `reticular markings`, `reticular pattern` |
| `lobar_segmental_collapse` | 32 | `lobar atelectasis`, `lobar collapse`, `segmental atelectasis` |
| `lung_lesion` | 6 | `lung lesion`, `pulmonary lesion`, `parenchymal lesion` |
| `lung_nodule_or_mass` | 21 | `lung nodule`, `lung nodules`, `pulmonary nodule` |
| `lung_opacity` | 8 | `lung opacity`, `lung opacities`, `pulmonary opacity` |
| `non_acute_rib_fracture` | 15 | `healed rib fracture`, `healed rib fractures`, `old rib fracture` |
| `nonsurgical_internal_foreign_body` | 29 | `foreign body`, `foreign bodies`, `ingested foreign body` |
| `other_hernia` | 4 | `diaphragmatic hernia`, `bochdalek hernia`, `morgagni hernia` |
| `pacemaker_electronic_cardiac_device_or_wires` | 26 | `pacemaker`, `dual chamber pacemaker`, `single chamber pacemaker` |
| `peribronchial_cuffing` | 6 | `peribronchial cuffing`, `peribronchial thickening`, `peribronchial markings` |
| `pleural_effusion` | 24 | `pleural effusion`, `pleural fluid`, `pleural collection` |
| `pleural_other` | 6 | `pleural abnormality`, `pleural disease`, `pleural lesion` |
| `pleural_thickening` | 7 | `pleural thickening`, `thickened pleura`, `pleural plaque` |
| `pneumomediastinum` | 5 | `pneumomediastinum`, `mediastinal air`, `mediastinal emphysema` |
| `pneumonia` | 10 | `pneumonia`, `bronchopneumonia`, `atypical pneumonia` |
| `pneumoperitoneum` | 6 | `pneumoperitoneum`, `free intraperitoneal air`, `free abdominal air` |
| `pneumothorax` | 12 | `pneumothorax`, `ptx`, `air pleural space` |
| `pulmonary_artery_enlargement` | 20 | `enlarged pulmonary arteries`, `pulmonary artery dilation`, `pulmonary artery dilatation` |
| `pulmonary_congestion_pulmonary_venous_congestion` | 19 | `congestion`, `pulmonary congestion`, `pulmonary venous congestion` |
| `pulmonary_edema` | 8 | `pulmonary edema`, `pulmonary oedema`, `interstitial pulmonary edema` |
| `pulmonary_fibrosis` | 15 | `pulmonary fibrosis`, `lung fibrosis`, `fibrotic change` |
| `shoulder_dislocation` | 19 | `glenohumeral joint dislocation`, `glenohumeral dislocation`, `shoulder dislocation` |
| `subcutaneous_emphysema` | 16 | `subcutaneous emphysema`, `subcutaneous gas`, `subcutaneous air` |
| `support_devices_generic` | 6 | `support device`, `support devices`, `hardware` |
| `tortuous_aorta` | 10 | `tortuous aorta`, `aortic tortuosity`, `ectatic aorta` |
| `tracheal_deviation` | 15 | `tracheal deviation`, `trachea deviated`, `deviated trachea` |
| `tracheostomy_tube` | 4 | `tracheostomy tube`, `tracheostomy`, `trach tube` |
| `whole_lung_or_majority_collapse` | 21 | `complete right side atelectasis`, `total right lung collapse`, `complete left side atelectasis` |

A few labels also carry an `exclude` clause that vetoes false positives, e.g.
`pleural_effusion` excludes `pericardial effusion`.

---

## Tuning knobs

### Uncertainty policy

Both `label_study` and `label_reports` accept:

- **`apply_exclude=False`** — disable the per-label exclude clauses (default is `True`).
- **`uncertainty_policy="keep" | "as_positive" | "as_negative" | "drop"`** —
  what to do with `uncertain` per-seed statuses. Default keeps them as a
  separate status; the others map them to present / absent / drop them
  entirely.

### RadGraph cache

Running RadGraph is the slowest step. Save annotations once and reload them on
subsequent runs without touching the GPU:

```bash
# First run: save annotations
radlabels label --file reports.json --save-radgraph-cache cache.json --out labels.json

# Later runs: skip inference entirely
radlabels label --file reports.json --radgraph-cache cache.json --out labels.json
```

Cache files include a `_meta` provenance header (radlabels version, alias version,
timestamp) so cached outputs remain auditable.

### Custom alias dictionaries

Replace the built-in dictionary with your own for CLI or Python use:

```bash
radlabels label --file reports.json --radgraph-cache cache.json \
    --custom-aliases my_aliases.json --out labels.json
```

```python
from radlabels import label_reports, validate_aliases

my_dict = {
    "my_finding": {"aliases": ["ground glass", "hazy opacity"], "exclude": []},
}
errors = validate_aliases(my_dict)   # check schema before use
results = label_reports(texts, aliases=my_dict)
```

The dictionary schema mirrors `ALIASES` — each key maps to `{"aliases": [...], "exclude": [...]}`.
`validate_aliases()` returns schema errors and duplicate-phrase warnings before running.

### Coarse-grained labels

`PARENT_MAP` maps all 49 leaf labels to 10 coarse parent groups, enabling hierarchical
aggregation without a separate lookup table:

```python
from radlabels import PARENT_MAP, label_study

labels, _, _ = label_study(anno)
coarse = {}
for leaf, status in labels.items():
    parent = PARENT_MAP[leaf]
    if parent not in coarse or status == "definitely present":
        coarse[parent] = status
```

See `examples/04_fine_to_coarse.py` for rule-based and score-based aggregation helpers.

---

## Citation

If you use this in academic work, please cite:

```bibtex
@article{delbrouck2026radlabels,
  title  = {Fast, Free, Accurate, Unlimited, and Controllable Radiological Labels},
  author = {Delbrouck, Jean-Benoit},
  year   = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE). The bundled corpus is released under the same
license.
