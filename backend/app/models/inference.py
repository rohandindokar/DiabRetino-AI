"""
Inference engine: loads trained PyTorch checkpoints, runs classification,
and assembles a clinical report with Grad-CAM explainability.

Models are stored in backend/weights/<name>.pth. If a checkpoint is missing
the engine falls back to a freshly-initialized (untrained) architecture so the
API still returns well-formed responses during development.
"""
import os
import time
import logging
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image

from app.core.config import CLASSES, CLASS_DISPLAY_NAMES, SEVERITY_METADATA
from app.models.architectures import build_model, MODEL_REGISTRY
from app.models.preprocessing import preprocess_image
from app.models.gradcam import generate_attention_artifacts

logger = logging.getLogger("diabretino.inference")


class DRInferenceEngine:
    def __init__(self, weights_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.models: Dict[str, torch.nn.Module] = {}
        self.model_loaded_from_checkpoint: Dict[str, bool] = {}
        self.active_model_name = "hybrid_vit_cnn"
        self._initialize()

    def _initialize(self):
        logger.info("Initializing DR engine on %s", self.device)
        for name in MODEL_REGISTRY:
            model = build_model(name, num_classes=5, pretrained=False).to(self.device)
            model.eval()
            self.models[name] = model
            self.model_loaded_from_checkpoint[name] = self._load_checkpoint(name, model)

    def _load_checkpoint(self, name: str, model) -> bool:
        path = os.path.join(self.weights_dir, f"{name}.pth")
        if not os.path.isfile(path):
            logger.info("No checkpoint for %s (using random init)", name)
            return False
        try:
            state = torch.load(path, map_location=self.device)
            # accept either a raw state_dict or our checkpoint dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint %s", path)
            return True
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)
            return False

    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes, model_name: str = "hybrid_vit_cnn",
                apply_enhancement: bool = True) -> Dict[str, Any]:
        start = time.time()
        if model_name not in self.models:
            model_name = "hybrid_vit_cnn"
        model = self.models[model_name]
        model.eval()

        prep = preprocess_image(image_bytes, apply_enhancement)
        input_tensor = prep["tensor"].to(self.device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            predicted = int(np.argmax(probs))
            confidence = float(probs[predicted])

        target_layer = model.get_last_conv_layer()
        if target_layer is not None:
            artifacts = generate_attention_artifacts(
                model, target_layer, input_tensor,
                prep["original_pil"], predicted)
        else:
            # ViT has no conv layer; reuse the hybrid's CNN branch for the map
            fallback = self.models["hybrid_vit_cnn"]
            artifacts = generate_attention_artifacts(
                fallback, fallback.get_last_conv_layer(),
                input_tensor, prep["original_pil"], predicted)

        meta = SEVERITY_METADATA[predicted]
        latency_ms = round((time.time() - start) * 1000, 1)

        return {
            "success": True,
            "prediction": {
                "class_id": predicted,
                "class_name": meta["name"],
                "class_key": meta["key"],
                "stage": meta["stage"],
                "confidence": round(confidence, 4),
                "confidence_percentage": round(confidence * 100, 2),
                "risk_level": meta["risk_level"],
                "urgency": meta["urgency"],
                "laser_required": meta["laser_required"],
                "badge_color": meta["badge_color"],
                "badge_bg": meta["badge_bg"],
            },
            "clinical_guidelines": {
                "icdr_criteria": meta["icdr_criteria"],
                "clinical_findings": meta["clinical_findings"],
                "recommended_action": meta["recommended_action"],
            },
            "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(5)},
            "probabilities_breakdown": [
                {
                    "id": i,
                    "key": CLASSES[i],
                    "name": CLASS_DISPLAY_NAMES[i],
                    "probability": round(float(probs[i]), 4),
                    "percentage": round(float(probs[i]) * 100, 2),
                    "color": SEVERITY_METADATA[i]["badge_color"],
                } for i in range(5)
            ],
            "visualizations": {
                "original_image": prep["original_base64"],
                "preprocessed_image": prep["processed_base64"],
                "heatmap": artifacts["heatmap_base64"],
                "overlay": artifacts["overlay_base64"],
            },
            "model_info": {
                "active_model": model_name,
                "architecture_display": MODEL_REGISTRY[model_name]["name"],
                "device": str(self.device),
                "latency_ms": latency_ms,
                "loaded_from_checkpoint": self.model_loaded_from_checkpoint.get(model_name, False),
            },
        }
