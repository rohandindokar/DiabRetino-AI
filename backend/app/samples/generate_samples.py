"""
Populate the reference-sample gallery with one real APTOS fundus image per
severity class (0-4). Falls back to a synthetic generator if the APTOS train
images are not present (e.g. inside a Docker build without the dataset).

Run locally with the dataset to get real reference images:
    python app/samples/generate_samples.py
"""
import os
import shutil
import logging

import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("samples")

HERE = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "data"))
TRAIN_IMAGES = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# output filename -> stage id
OUT_FILES = {
    "stage_0_no_dr.png": 0,
    "stage_1_mild.png": 1,
    "stage_2_moderate.png": 2,
    "stage_3_severe.png": 3,
    "stage_4_proliferative.png": 4,
}


def pick_from_aptos():
    """Copy one image per class from the APTOS train split into samples/."""
    if not os.path.isfile(TRAIN_CSV):
        return False
    df = pd.read_csv(TRAIN_CSV)
    for out_name, stage in OUT_FILES.items():
        # first image of this class that actually exists on disk
        subset = df[df["diagnosis"] == stage]
        placed = False
        for code in subset["id_code"].tolist():
            for ext in (".png", ".jpg"):
                src = os.path.join(TRAIN_IMAGES, code + ext)
                if os.path.isfile(src):
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        im = im.resize((512, 512), Image.LANCZOS)
                        im.save(os.path.join(HERE, out_name), "PNG")
                    log.info("[+] real sample -> %s  (class %d, id %s)",
                             out_name, stage, code)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            log.warning("[!] no image found for class %d", stage)
    return True


if __name__ == "__main__":
    if not pick_from_aptos():
        log.info("APTOS dataset not found - regenerating synthetic samples")
        from synthetic_samples import generate_all_synthetic
        generate_all_synthetic(HERE)
    log.info("Reference samples ready in %s", HERE)
