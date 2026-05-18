"""Label a single report with the Python API."""
from radlabels import label_reports

text = (
    "FINDINGS:\n"
    "Cardiac silhouette is mildly enlarged. Small left pleural effusion.\n"
    "No pneumothorax. The lungs are otherwise clear.\n"
    "IMPRESSION:\n"
    "Mild cardiomegaly with a small left pleural effusion."
)

[result] = label_reports([text])

print("Labels:")
for disease, status in sorted(result.labels.items()):
    print(f"  {disease:30s}  {status}")

print("\nMatches:")
for m in result.matches:
    print(f"  disease={m['disease']:24s} alias={m['alias']:30s} status={m['label']}")
