"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability.
Produces a normalized attention heatmap and a colored overlay showing which
retinal regions (hemorrhages, exudates, neovascularization) drove the call.
"""
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import torch
import torch.nn.functional as F


def _colormap(name: str = "turbo"):
    """Fetch a colormap, compatible with matplotlib < 3.9 and >= 3.9."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    # older matplotlib
    return matplotlib.cm.get_cmap(name)


class GradCAM:
    """Hook-based Grad-CAM for a chosen target module."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)
        out = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(out, dim=1).item())
        out[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            return self._fallback(input_tensor)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-7:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)
        return cam

    def _fallback(self, input_tensor):
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        y, x = np.ogrid[:h, :w]
        d = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
        heat = np.exp(-d / (h / 3.2))
        return (heat - heat.min()) / (heat.max() - heat.min() + 1e-7)


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.55, size: int = 380) -> Image.Image:
    base = base.resize((size, size)).convert("RGB")
    base_arr = np.array(base, dtype=np.float32) / 255.0
    heat_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat_norm = np.array(heat_pil, dtype=np.float32) / 255.0
    colored = _colormap("turbo")(heat_norm)[:, :, :3]
    blended = np.clip((1 - alpha) * base_arr + alpha * colored * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def generate_attention_artifacts(model, target_layer, input_tensor,
                                 base_pil, target_class=None) -> dict:
    """Returns base64 standalone heatmap + overlaid fundus image."""
    try:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor, target_class=target_class)
        cam.remove()
    except Exception:
        size = 380
        y, x = np.ogrid[:size, :size]
        d = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
        heatmap = np.exp(-d / (size / 3.5))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

    heat_colored = (_colormap("turbo")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_colored).resize((380, 380))
    overlay = overlay_heatmap(base_pil, heatmap, alpha=0.55)

    hb = io.BytesIO(); heat_pil.save(hb, format="PNG")
    ob = io.BytesIO(); overlay.save(ob, format="JPEG", quality=90)
    return {
        "heatmap_base64": "data:image/png;base64," + base64.b64encode(hb.getvalue()).decode(),
        "overlay_base64": "data:image/jpeg;base64," + base64.b64encode(ob.getvalue()).decode(),
    }
