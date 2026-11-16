"""
Generate a human-readable metrics report (reports/REPORT.md) from the
machine-readable metrics_report.json produced by train.py.

    python make_report.py
"""
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BACKEND_DIR / "reports"
ROOT_REPORT = BACKEND_DIR.parent / "reports"

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def load_report():
    p = REPORTS_DIR / "metrics_report.json"
    if not p.exists():
        raise FileNotFoundError("Run `python train.py` first to generate metrics_report.json")
    return json.loads(p.read_text())


def cm_as_table(cm, classes):
    lines = ["| actual \\ predicted | " + " | ".join(classes) + " |",
             "| " + " --- " * (len(classes) + 1) + " |"]
    for i, row in enumerate(cm):
        cells = [str(v) for v in row]
        lines.append(f"| **{classes[i]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    data = load_report()
    models = data["models"]
    flagship = next((m for m in models if m["model"] == "hybrid_vit_cnn"), models[0])

    out = []
    out.append("# Diabetic Retinopathy Severity Model — Evaluation Report\n")
    out.append(f"*Generated: {data['generated_at']}  ·  Device: {data['device']}  ·  "
               f"Dataset: {data['dataset']} ({data['n_train']} train images)*\n")

    out.append("## Class distribution (training set)\n")
    out.append("| Class | Count | % |")
    out.append("| --- | --- | --- |")
    for c in CLASS_NAMES:
        out.append(f"| {c} | {data['class_distribution'].get(c, 0)} | "
                   f"{data['class_distribution_pct'].get(c, 0)}% |")
    out.append("")

    out.append("## Headline metrics (flagship: " + flagship["model"] + ")\n")
    out.append(f"- **Accuracy: {flagship['accuracy']*100:.1f}%**")
    out.append(f"- Macro F1: {flagship['macro_f1']*100:.1f}%   ·   "
               f"Weighted F1: {flagship['weighted_f1']*100:.1f}%")
    out.append(f"- Macro Precision: {flagship['precision_macro']*100:.1f}%   ·   "
               f"Macro Recall: {flagship['recall_macro']*100:.1f}%")
    out.append(f"- Evaluated on {flagship['n_samples']} held-out / training samples\n")

    out.append("### Per-class performance (flagship)\n")
    out.append("| Class | Precision | Recall | F1 | Support |")
    out.append("| --- | --- | --- | --- | --- |")
    for c in CLASS_NAMES:
        pc = flagship["per_class"][c]
        out.append(f"| {c} | {pc['precision']*100:.1f}% | {pc['recall']*100:.1f}% | "
                   f"{pc['f1']*100:.1f}% | {pc['support']} |")
    out.append("\n*Recall is the \"how many times it was right\" per stage — the share of "
               "true stage-X scans that the model correctly labelled as stage X.*\n")

    out.append("### Confusion matrix (rows = actual, columns = predicted)\n")
    out.append(cm_as_table(flagship["confusion_matrix"], CLASS_NAMES))
    out.append("")

    out.append("## All architectures\n")
    out.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    out.append("| --- | --- | --- | --- |")
    for m in models:
        out.append(f"| {m['model']} | {m['accuracy']*100:.1f}% | "
                   f"{m['macro_f1']*100:.1f}% | {m['weighted_f1']*100:.1f}% |")
    out.append("")

    out.append("## Notes\n")
    for n in data.get("notes", []):
        out.append(f"- {n}")
    out.append("\n---\n*These metrics are measured on the training split and therefore "
                "represent an upper bound on generalisation. They are intended to "
                "demonstrate that the pipeline produces meaningful, class-discriminating "
                "predictions — not as a validated clinical claim.*")

    text = "\n".join(out)
    # write to both backend/reports and repo-root reports/
    (REPORTS_DIR / "REPORT.md").write_text(text)
    ROOT_REPORT.mkdir(exist_ok=True)
    (ROOT_REPORT / "REPORT.md").write_text(text)
    print(f"[+] Wrote {REPORTS_DIR/'REPORT.md'} and {ROOT_REPORT/'REPORT.md'}")
    print(text[:600] + "\n...")


if __name__ == "__main__":
    main()
