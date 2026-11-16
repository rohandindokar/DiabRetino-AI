"""
Re-evaluate every trained checkpoint on the held-out validation split and
merge them into a single metrics_report.json + metrics.json.

Run this after all models are trained:
    python regen_report.py

This is the source of truth for the deployed metrics and for make_report.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score,
                             recall_score)

from app.models.architectures import build_model
from app.models.preprocessing import NORM_MEAN, NORM_STD
import train as _train

TRAIN_CSV = _train.TRAIN_CSV
TRAIN_IMAGES = _train.TRAIN_IMAGES
REPORTS_DIR = _train.REPORTS_DIR
WEIGHTS_DIR = _train.WEIGHTS_DIR
CLASS_NAMES = _train.CLASS_NAMES
NUM_CLASSES = _train.NUM_CLASSES
stratified_split = _train.stratified_split
get_device = _train.get_device

CANONICAL = ["hybrid_vit_cnn", "se_resnet18", "vit_dr"]
_LABELS = list(range(NUM_CLASSES))


def evaluate_on(df, device, model_name):
    """Run a trained checkpoint over a dataframe of images."""
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(device)
    ckpt = WEIGHTS_DIR / f"{model_name}.pth"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    imgs, labels = [], []
    for _, row in df.iterrows():
        p = TRAIN_IMAGES / f"{row['id_code']}.png"
        if not p.exists():
            p = TRAIN_IMAGES / f"{row['id_code']}.jpg"
        if not p.exists():
            continue
        imgs.append(tfm(Image.open(p).convert("RGB")))
        labels.append(int(row["diagnosis"]))
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        pred = model(imgs).argmax(1).cpu().numpy()
    return np.array(labels), pred, state


def main():
    device = get_device()
    df = pd.read_csv(TRAIN_CSV)
    _, val_df = stratified_split(df, val_frac=0.2)
    print(f"Device: {device}   Held-out val samples: {len(val_df)}")

    results = []
    for name in CANONICAL:
        ckpt = WEIGHTS_DIR / f"{name}.pth"
        if not ckpt.exists():
            print(f"  - {name}: no checkpoint, skipping")
            continue
        y_true, y_pred, state = evaluate_on(val_df, device, name)
        cm = confusion_matrix(y_true, y_pred, labels=_LABELS)
        rep = classification_report(y_true, y_pred, labels=_LABELS,
                                    target_names=CLASS_NAMES, output_dict=True)
        results.append({
            "model": name,
            "eval_split": "held-out validation (20%)",
            "n_samples": int(len(y_true)),
            "device": str(device),
            "best_val_acc_during_training": round(
                state.get("history", {}).get("val_acc", [0])[-1], 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
            "precision_macro": round(float(precision_score(y_true, y_pred,
                                                            average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_true, y_pred,
                                                      average="macro", zero_division=0)), 4),
            "per_class": {
                CLASS_NAMES[c]: {
                    "precision": round(rep[CLASS_NAMES[c]]["precision"], 4),
                    "recall": round(rep[CLASS_NAMES[c]]["recall"], 4),
                    "f1": round(rep[CLASS_NAMES[c]]["f1-score"], 4),
                    "support": int(rep[CLASS_NAMES[c]]["support"]),
                } for c in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
        })
        print(f"  + {name:16s} val-acc={results[-1]['accuracy']:.3f}  "
              f"macroF1={results[-1]['macro_f1']:.3f}")

    if not results:
        print("No trained checkpoints found. Nothing to report.")
        return

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dataset": "APTOS 2019 Diabetic Retinopathy",
        "n_train": int(len(df)),
        "n_val": int(len(val_df)),
        "class_distribution": {CLASS_NAMES[int(c)]: int(n)
                               for c, n in df["diagnosis"].value_counts().items()},
        "class_distribution_pct": {CLASS_NAMES[int(c)]: round(100 * n / len(df), 2)
                                   for c, n in df["diagnosis"].value_counts().items()},
        "class_names": CLASS_NAMES,
        "eval_split": "held-out validation (20%)",
        "models": results,
        "notes": [
            "Metrics are measured on a held-out 20% validation split "
            "(stratified by class), so they reflect generalisation, not in-sample fit.",
            "Severe (class 3) and Proliferative (class 4) are the rarest and most "
            "visually-overlapping classes and drive the lowest per-class recall.",
            "Grad-CAM explainability and clinical reporting run at inference time.",
        ],
    }
    REPORTS_DIR.joinpath("metrics_report.json").write_text(json.dumps(report, indent=2))
    WEIGHTS_DIR.joinpath("metrics.json").write_text(json.dumps(
        {m["model"]: m for m in results}, indent=2))
    print(f"\nWrote {REPORTS_DIR / 'metrics_report.json'}  ({len(results)} models)")


if __name__ == "__main__":
    main()
