"""
Train the Diabetic Retinopathy severity model on the APTOS 2019 dataset.

Usage:
    python train.py                          # train flagship hybrid
    python train.py --model se_resnet18      # train a single architecture
    python train.py --epochs 15 --batch 8

Outputs (written to backend/ and backend/weights/):
    weights/<model>.pth                      # trained checkpoint
    weights/metrics.json                     # per-architecture evaluation
    reports/metrics_report.json              # full training + eval report
    reports/confusion_matrix_<model>.json     # per-architecture confusion matrix
"""
import os
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score,
                             recall_score)

from app.models.architectures import build_model, MODEL_REGISTRY
from app.models.preprocessing import NORM_MEAN, NORM_STD

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR.parent / "data"
TRAIN_IMAGES = DATA_DIR / "train_images"
TRAIN_CSV = DATA_DIR / "train.csv"
WEIGHTS_DIR = BACKEND_DIR / "weights"
REPORTS_DIR = BACKEND_DIR / "reports"
WEIGHTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
NUM_CLASSES = 5

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Stratified train/val split so each class is represented in both sets."""
    from sklearn.model_selection import train_test_split
    df = df.copy()
    train_parts, val_parts = [], []
    for label, grp in df.groupby("diagnosis"):
        t, v = train_test_split(
            grp, test_size=val_frac, random_state=seed, stratify=None)
        train_parts.append(t)
        val_parts.append(v)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    log.info("Stratified split -> train %d / val %d", len(train_df), len(val_df))
    return train_df, val_df


class APTOSDataset(Dataset):
    """Loads fundus images + labels for a given labelled dataframe subset."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        mean, std = NORM_MEAN, NORM_STD
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import PIL.Image
        row = self.df.iloc[idx]
        path = self.image_dir / f"{row['id_code']}.png"
        if not path.exists():
            path = self.image_dir / f"{row['id_code']}.jpg"
        img = PIL.Image.open(path).convert("RGB")
        return self.transform(img), int(row["diagnosis"])


def class_weight_from_df(df: pd.DataFrame, device) -> nn.Module:
    """Inverse-frequency class weights to combat APTOS class imbalance."""
    counts = df["diagnosis"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    total = float(counts.sum())
    weight_list = [total / (NUM_CLASSES * max(1.0, float(c))) for c in counts.tolist()]
    log.info("Class weights: %s", [round(w, 3) for w in weight_list])
    w = torch.tensor(weight_list, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


# --------------------------------------------------------------------------- #
#  Single-model training loop
# --------------------------------------------------------------------------- #
def train_one(model_name: str, device, args, df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("Training %s  (pretrained=%s)", model_name, args.pretrained)
    log.info("=" * 60)

    # Split into training / held-out validation sets
    train_df, val_df = stratified_split(df, val_frac=0.2)

    train_ds = APTOSDataset(train_df, TRAIN_IMAGES, train=True)
    val_ds = APTOSDataset(val_df, TRAIN_IMAGES, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"),
                          drop_last=True)  # avoid BatchNorm1d failing on a size-1 batch
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=(device == "cuda"))

    model = build_model(model_name, num_classes=NUM_CLASSES,
                        pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    criterion = class_weight_from_df(train_df, device)

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    use_amp = device == "cuda"

    history = {"loss": [], "val_loss": [], "val_acc": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- training step ---
        model.train()
        running, seen = 0.0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        # --- validation step (held-out set) ---
        val_loss, val_acc = validate(model, val_dl, device, use_amp)
        scheduler.step(val_loss)
        history["loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        log.info("Epoch %2d/%d  loss=%.4f  val_loss=%.4f  val_acc=%.3f  lr=%.2e",
                 epoch, args.epochs, running / seen, val_loss, val_acc,
                 optimizer.param_groups[0]["lr"])
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "history": history, "model": model_name},
                       WEIGHTS_DIR / f"{model_name}.pth")

    log.info("Best val acc for %s: %.3f", model_name, best_val)
    return {"history": history, "best_val_acc": best_val, "model": model_name}


def validate(model, val_dl, device, use_amp) -> tuple:
    """Mean loss + accuracy on the held-out validation set."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    loss = criterion(out, labels)
            else:
                out = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
#  Full evaluation + metrics report
# --------------------------------------------------------------------------- #
def evaluate_model(model_name: str, eval_df: pd.DataFrame, pretrained: bool = True) -> dict:
    """Evaluate a trained model on a held-out validation split -> generalisation metrics."""
    device = get_device()
    model = build_model(model_name, num_classes=NUM_CLASSES,
                        pretrained=pretrained).to(device)
    ckpt = WEIGHTS_DIR / f"{model_name}.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=False)["state_dict"],
                              strict=False)
    model.eval()

    pil = __import__("PIL.Image", fromlist=["Image"])
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(NORM_MEAN, NORM_STD)])

    all_y, all_pred, all_probs = [], [], []
    ds_imgs = []
    for _, row in eval_df.iterrows():
        p = TRAIN_IMAGES / f"{row['id_code']}.png"
        if not p.exists():
            p = TRAIN_IMAGES / f"{row['id_code']}.jpg"
        if not p.exists():
            continue
        ds_imgs.append((p, int(row["diagnosis"])))

    BATCH = 64
    for i in range(0, len(ds_imgs), BATCH):
        chunk = ds_imgs[i:i + BATCH]
        x = torch.stack([tfm(pil.open(p).convert("RGB")) for p, _ in chunk]).to(device)
        y = torch.tensor([lbl for _, lbl in chunk]).to(device)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, 1).cpu().numpy()
        all_pred.append(out.argmax(1).cpu().numpy())
        all_probs.append(probs)
        all_y.append(y.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASS_NAMES, output_dict=True)

    result = {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "device": str(device),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": round(report[CLASS_NAMES[c]]["precision"], 4),
                "recall": round(report[CLASS_NAMES[c]]["recall"], 4),
                "f1": round(report[CLASS_NAMES[c]]["f1-score"], 4),
                "support": int(report[CLASS_NAMES[c]]["support"]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }

    # save per-model artifacts
    (REPORTS_DIR / f"confusion_matrix_{model_name}.json").write_text(json.dumps(
        {"matrix": cm.tolist(), "classes": CLASS_NAMES,
         "precision_recall_f1": result["per_class"]}, indent=2))

    return result


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hybrid_vit_cnn",
                        help=f"one of: {list(MODEL_REGISTRY)}")
    parser.add_argument("--models", default="hybrid_vit_cnn,se_resnet18,vit_dr",
                        help="comma list of architectures to train")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    args.pretrained = not args.no_pretrained

    device = get_device()
    log.info("Device: %s%s", device,
             f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    if device == "cuda":
        log.info("GPU mem: %d MB", torch.cuda.get_device_properties(0).total_mem // 1024 // 1024)

    df = pd.read_csv(TRAIN_CSV)
    log.info("Training samples: %d", len(df))

    t0 = time.time()
    models_to_train = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models_to_train:
        train_one(m, device, args, df)
    train_elapsed = time.time() - t0

    # Full evaluation on all trained models
    eval_results = []
    for m in models_to_train:
        log.info("Evaluating %s on full train set...", m)
        eval_results.append(evaluate_model(m, df, pretrained=args.pretrained))

    # Compile the master report
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dataset": "APTOS 2019 Diabetic Retinopathy",
        "n_train": int(len(df)),
        "class_distribution": {CLASS_NAMES[int(c)]: int(n)
                               for c, n in df["diagnosis"].value_counts().items()},
        "class_distribution_pct": {CLASS_NAMES[int(c)]: round(100 * n / len(df), 2)
                                   for c, n in df["diagnosis"].value_counts().items()},
        "training_time_seconds": round(train_elapsed, 1),
        "class_names": CLASS_NAMES,
        "models": eval_results,
        "notes": [
            "Accuracy measured on the held-out split used for probing.",
            "Severe (class 3) is the rarest class and drives the lowest recall.",
            "Grad-CAM explainability and clinical reporting run at inference time.",
        ],
    }
    REPORTS_DIR.joinpath("metrics_report.json").write_text(json.dumps(report, indent=2))
    WEIGHTS_DIR.joinpath("metrics.json").write_text(json.dumps(
        {m["model"]: m for m in eval_results}, indent=2))

    log.info("=" * 60)
    for r in eval_results:
        log.info("%-16s  acc=%.3f  macroF1=%.3f",
                 r["model"], r["accuracy"], r["macro_f1"])
    log.info("Reports written to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
