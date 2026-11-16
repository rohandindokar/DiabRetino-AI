"""
Retinal fundus preprocessing:
  - auto-circular aperture crop (removes black camera margins)
  - Ben Graham local color-subtraction contrast enhancement
  - PyTorch normalization into a (1, 3, 224, 224) tensor
"""
import io
import base64
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms

# ImageNet normalization (matches pretrained ResNet / ViT expectations)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def crop_fundus_circle(img: Image.Image, tol: int = 18) -> Image.Image:
    """Crop the black rectangular margin around a circular fundus photo."""
    gray = np.array(img.convert("L"))
    mask = gray > tol
    if not mask.any():
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h, w = gray.shape
    pad = int(min(ymax - ymin, xmax - xmin) * 0.02)
    ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)
    xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
    return img.crop((xmin, ymin, xmax, ymax))


def ben_graham_enhancement(img: Image.Image, sigma: float = 9.0) -> Image.Image:
    """Ben Graham's local color subtraction (4*(I - GaussianBlur(I)) + 128)."""
    work = img.resize((512, 512), Image.Resampling.BILINEAR)
    blur = work.filter(ImageFilter.GaussianBlur(radius=sigma))
    orig = np.array(work, dtype=np.float32)
    b = np.array(blur, dtype=np.float32)
    enhanced = np.clip(4.0 * orig - 4.0 * b + 128.0, 0, 255).astype(np.uint8)

    # circular mask to clean the corners
    h, w, _ = enhanced.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(cy, cx) * 0.96
    m = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    enhanced[~m] = 0
    return Image.fromarray(enhanced)


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def preprocess_image(image_bytes: bytes, apply_enhancement: bool = False) -> dict:
    """Returns a dict with the model tensor, original/processed PIL images,
    and base64 thumbnails for the GUI side-by-side renderer.

    When apply_enhancement is False the image is resized+normalised directly,
    matching the training pipeline (no circular crop, no Ben Graham) - this is
    the mode that reproduces the model's training-time input distribution.
    """
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if apply_enhancement:
        cropped = crop_fundus_circle(original)
        processed = ben_graham_enhancement(cropped)
    else:
        # Match training exactly: plain square resize, no crop, no enhancement.
        processed = original.resize((512, 512), Image.Resampling.BILINEAR)

    tensor = inference_transform(processed).unsqueeze(0)

    return {
        "tensor": tensor,
        "original_pil": original,
        "processed_pil": processed,
        "original_base64": image_to_base64(original.resize((380, 380))),
        "processed_base64": image_to_base64(processed.resize((380, 380))),
    }
