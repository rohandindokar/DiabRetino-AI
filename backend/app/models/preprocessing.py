"""
Retinal Image Preprocessing Utilities for Diabetic Retinopathy.
Implements:
1. Auto-circle cropping (removes black camera margins)
2. Ben Graham's local color subtraction / contrast enhancement
3. PyTorch normalization & resizing
"""
import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torchvision.transforms as transforms


# Standard PyTorch normalization for ViT and ResNet backbones
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def crop_fundus_circle(img: Image.Image, tol: int = 15) -> Image.Image:
    """
    Crops black rectangular margins around circular retinal fundus photograph.
    """
    gray = img.convert('L')
    gray_arr = np.array(gray)
    mask = gray_arr > tol

    if not mask.any():
        return img  # Entirely dark, return as-is

    # Get bounding box of retinal fundus
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Add small margin padding
    h, w = gray_arr.shape
    pad = int(min(ymax - ymin, xmax - xmin) * 0.02)
    ymin = max(0, ymin - pad)
    ymax = min(h, ymax + pad)
    xmin = max(0, xmin - pad)
    xmax = min(w, xmax + pad)

    return img.crop((xmin, ymin, xmax, ymax))


def apply_ben_graham_enhancement(img: Image.Image, sigma: float = 10.0) -> Image.Image:
    """
    Ben Graham's Retinal Color Normalization:
    Subtracts local average color (via Gaussian Blur) to eliminate illumination gradients
    and heighten visibility of microaneurysms, hemorrhages, and exudates.
    Formula: output = 4 * image - 4 * gaussian_blur(image) + 128
    """
    # Resize to working resolution for consistent blur radius
    work_img = img.resize((512, 512), Image.Resampling.BILINEAR)
    blur_img = work_img.filter(ImageFilter.GaussianBlur(radius=sigma))

    arr_orig = np.array(work_img, dtype=np.float32)
    arr_blur = np.array(blur_img, dtype=np.float32)

    enhanced = 4.0 * arr_orig - 4.0 * arr_blur + 128.0
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Optional circular mask to clean outside edges
    h, w, _ = enhanced.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = min(center_y, center_x) * 0.96
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius ** 2)

    enhanced[~mask] = 0
    return Image.fromarray(enhanced)


def preprocess_image(image_bytes: bytes, apply_enhancement: bool = True):
    """
    Full preprocessing pipeline:
    Reads bytes -> Crops fundus circle -> Enhances with Ben Graham filter -> Prepares Tensor & Base64.
    """
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cropped_img = crop_fundus_circle(original_img)

    if apply_enhancement:
        processed_img = apply_ben_graham_enhancement(cropped_img)
    else:
        processed_img = cropped_img.resize((512, 512), Image.Resampling.BILINEAR)

    # Tensor for deep learning models (224x224)
    model_tensor = inference_transform(processed_img).unsqueeze(0)

    # Base64 representations for GUI side-by-side rendering
    orig_b64 = image_to_base64(original_img.resize((380, 380)))
    proc_b64 = image_to_base64(processed_img.resize((380, 380)))

    return {
        "tensor": model_tensor,
        "original_pil": original_img,
        "processed_pil": processed_img,
        "original_base64": orig_b64,
        "processed_base64": proc_b64
    }


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Helper to convert PIL Image to base64 data URI."""
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"
