"""
Generates synthetic representative fundus images for the 5 Diabetic Retinopathy severity stages.
Used for immediate testing, evaluation, and GUI demo gallery.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_base_retina(width=512, height=512):
    """Generates standard retinal fundus orange-red gradient background."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2
    radius = int(width * 0.46)

    # Fundus base circle with radial lighting
    for r in range(radius, 0, -2):
        factor = r / radius
        # Warm retinal tones: orange-red gradient fading slightly near periphery
        red = int(180 + 55 * (1 - factor))
        green = int(60 + 40 * (1 - factor))
        blue = int(20 + 20 * (1 - factor))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(red, green, blue))

    # Optic Disc (yellowish-orange oval towards nasal side)
    disc_x, disc_y = int(cx - radius * 0.42), cy
    disc_rx, disc_ry = int(radius * 0.16), int(radius * 0.20)
    draw.ellipse([disc_x - disc_rx, disc_y - disc_ry, disc_x + disc_rx, disc_y + disc_ry], fill=(245, 215, 120))
    # Optic cup
    draw.ellipse([disc_x - disc_rx//2, disc_y - disc_ry//2, disc_x + disc_rx//2, disc_y + disc_ry//2], fill=(255, 240, 180))

    # Macula (darker reddish-brown zone towards temporal side)
    macula_x, macula_y = int(cx + radius * 0.25), cy
    mac_r = int(radius * 0.20)
    for mr in range(mac_r, 0, -2):
        draw.ellipse([macula_x - mr, macula_y - mr, macula_x + mr, macula_y + mr], fill=(130, 35, 15))
    # Fovea centralis
    draw.ellipse([macula_x - 4, macula_y - 4, macula_x + 4, macula_y + 4], fill=(90, 20, 10))

    # Retinal Vascular Arcade (Major blood vessels emanating from disc)
    def draw_vessel_branch(points, width_start=7, color=(120, 20, 15)):
        for i in range(len(points) - 1):
            w = max(1, int(width_start * (1 - i / len(points))))
            draw.line([points[i], points[i+1]], fill=color, width=w)

    # Superior temporal arcade
    pts_sup = [
        (disc_x, disc_y),
        (disc_x + 30, disc_y - 50),
        (disc_x + 90, disc_y - 110),
        (disc_x + 180, disc_y - 140),
        (disc_x + 280, disc_y - 120),
        (disc_x + 350, disc_y - 80)
    ]
    draw_vessel_branch(pts_sup, width_start=6)

    # Inferior temporal arcade
    pts_inf = [
        (disc_x, disc_y),
        (disc_x + 30, disc_y + 50),
        (disc_x + 90, disc_y + 110),
        (disc_x + 180, disc_y + 140),
        (disc_x + 280, disc_y + 120),
        (disc_x + 350, disc_y + 80)
    ]
    draw_vessel_branch(pts_inf, width_start=6)

    # Nasal vessels
    draw_vessel_branch([(disc_x, disc_y), (disc_x - 50, disc_y - 60), (disc_x - 110, disc_y - 90)], width_start=4)
    draw_vessel_branch([(disc_x, disc_y), (disc_x - 50, disc_y + 60), (disc_x - 110, disc_y + 90)], width_start=4)

    # Soft Gaussian filter for natural biological blending
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img, (cx, cy, radius)


def generate_all_samples(output_dir: str):
    """Generates the 5 clinical stages and saves to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    stages = [
        ("stage_0_no_dr.jpg", 0),
        ("stage_1_mild_dr.jpg", 1),
        ("stage_2_moderate_dr.jpg", 2),
        ("stage_3_severe_dr.jpg", 3),
        ("stage_4_proliferative_dr.jpg", 4)
    ]

    for filename, stage in stages:
        img, (cx, cy, radius) = create_base_retina()
        draw = ImageDraw.Draw(img)

        if stage == 0:
            # Stage 0: Clean retina, no lesions
            pass

        elif stage == 1:
            # Stage 1: Mild DR - Isolated tiny microaneurysms (small dark red dots, 2-4px)
            for _ in range(8):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(radius * 0.2, radius * 0.7)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                draw.ellipse([lx - 2, ly - 2, lx + 2, ly + 2], fill=(60, 10, 8))

        elif stage == 2:
            # Stage 2: Moderate DR - Hard exudates (bright yellow wax) & micro-hemorrhages
            # Exudates (bright yellow spots around macula)
            for _ in range(16):
                angle = np.random.uniform(-0.5, 0.5)
                dist = np.random.uniform(radius * 0.3, radius * 0.6)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                draw.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=(255, 235, 110))
            # Blot hemorrhages
            for _ in range(14):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(radius * 0.2, radius * 0.8)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                draw.ellipse([lx - 4, ly - 4, lx + 4, ly + 4], fill=(55, 8, 6))

        elif stage == 3:
            # Stage 3: Severe NPDR - Dense 4-quadrant blot hemorrhages & cotton-wool patches
            for _ in range(48):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(radius * 0.15, radius * 0.85)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                r_size = np.random.randint(4, 9)
                draw.ellipse([lx - r_size, ly - r_size, lx + r_size, ly + r_size], fill=(45, 5, 5))
            # Cotton wool spots (fluffy white/grey infarcts)
            for _ in range(7):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(radius * 0.3, radius * 0.7)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                draw.ellipse([lx - 9, ly - 7, lx + 9, ly + 7], fill=(230, 230, 225))

        elif stage == 4:
            # Stage 4: Proliferative DR - Extensive neovascularization fronds, vitreous dark bleed
            # Vitreous / preretinal hemorrhage (large irregular dark pool)
            draw.polygon([
                (cx + 40, cy - 60),
                (cx + 120, cy - 90),
                (cx + 160, cy - 40),
                (cx + 130, cy + 20),
                (cx + 60, cy + 10)
            ], fill=(35, 3, 3))
            # Neovascularization (chaotic tangle of thin fragile new vessels)
            for _ in range(35):
                start_x = int(cx - radius * 0.4 + np.random.randint(-20, 20))
                start_y = int(cy + np.random.randint(-20, 20))
                end_x = start_x + np.random.randint(15, 60)
                end_y = start_y + np.random.randint(-40, 40)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(150, 10, 10), width=2)
            # Widespread severe hemorrhages
            for _ in range(60):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(radius * 0.1, radius * 0.88)
                lx = int(cx + dist * math.cos(angle))
                ly = int(cy + dist * math.sin(angle))
                r_size = np.random.randint(4, 11)
                draw.ellipse([lx - r_size, ly - r_size, lx + r_size, ly + r_size], fill=(40, 5, 5))

        save_path = os.path.join(output_dir, filename)
        img.save(save_path, "JPEG", quality=95)
        print(f"[+] Generated sample: {save_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    generate_all_samples(current_dir)
