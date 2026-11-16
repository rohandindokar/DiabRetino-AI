"""
Synthetic fundus-sample generator.
Used only as a fallback when the real APTOS dataset is unavailable
(e.g. inside a Docker build with no dataset mounted). Produces plausible
orange-red circular fundus images with stage-appropriate lesion overlays.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _base_retina(w=512, h=512):
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    radius = int(w * 0.46)
    for r in range(radius, 0, -2):
        f = r / radius
        red = int(185 + 55 * (1 - f))
        green = int(65 + 40 * (1 - f))
        blue = int(25 + 20 * (1 - f))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(red, green, blue))
    # optic disc
    dx, dy = int(cx - radius * 0.42), cy
    d.ellipse([dx - radius * .16, dy - radius * .20, dx + radius * .16, dy + radius * .20],
              fill=(245, 215, 120))
    # vessels
    for (pts, width) in [
        ([(dx, dy), (dx + 90, dy - 110), (dx + 280, dy - 80)], 5),
        ([(dx, dy), (dx + 90, dy + 110), (dx + 280, dy + 80)], 5),
    ]:
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=(120, 20, 15), width=width)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img, (cx, cy, radius)


def generate_all_synthetic(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(7)
    stages = {
        "stage_0_no_dr.png": 0, "stage_1_mild.png": 1,
        "stage_2_moderate.png": 2, "stage_3_severe.png": 3,
        "stage_4_proliferative.png": 4,
    }
    for fname, stage in stages.items():
        img, (cx, cy, radius) = _base_retina()
        d = ImageDraw.Draw(img)
        if stage == 1:
            for _ in range(8):
                a = np.random.uniform(0, 2 * math.pi); dist = np.random.uniform(radius * .2, radius * .7)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                d.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=(60, 10, 8))
        elif stage == 2:
            for _ in range(14):
                a = np.random.uniform(-.5, .5); dist = np.random.uniform(radius * .3, radius * .6)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                d.ellipse([lx - 4, ly - 4, lx + 4, ly + 4], fill=(255, 235, 110))
            for _ in range(12):
                a = np.random.uniform(0, 2 * math.pi); dist = np.random.uniform(radius * .2, radius * .8)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                d.ellipse([lx - 4, ly - 4, lx + 4, ly + 4], fill=(55, 8, 6))
        elif stage == 3:
            for _ in range(48):
                a = np.random.uniform(0, 2 * math.pi); dist = np.random.uniform(radius * .15, radius * .85)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                r = np.random.randint(4, 9)
                d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(45, 5, 5))
            for _ in range(6):
                a = np.random.uniform(0, 2 * math.pi); dist = np.random.uniform(radius * .3, radius * .7)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                d.ellipse([lx - 9, ly - 7, lx + 9, ly + 7], fill=(230, 230, 225))
        elif stage == 4:
            d.polygon([(cx + 40, cy - 60), (cx + 120, cy - 90), (cx + 160, cy - 40),
                       (cx + 130, cy + 20), (cx + 60, cy + 10)], fill=(35, 3, 3))
            for _ in range(35):
                sx = int(cx - radius * .4 + np.random.randint(-20, 20))
                sy = int(cy + np.random.randint(-20, 20))
                d.line([(sx, sy), (sx + np.random.randint(15, 60), sy + np.random.randint(-40, 40))],
                       fill=(150, 10, 10), width=2)
            for _ in range(60):
                a = np.random.uniform(0, 2 * math.pi); dist = np.random.uniform(radius * .1, radius * .88)
                lx, ly = int(cx + dist * math.cos(a)), int(cy + dist * math.sin(a))
                r = np.random.randint(4, 11)
                d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(40, 5, 5))
        img.save(os.path.join(out_dir, fname), "PNG")
        print(f"[+] synthetic sample -> {fname}")
