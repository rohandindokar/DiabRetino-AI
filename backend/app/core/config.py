"""
Configuration and Clinical Metadata for Diabetic Retinopathy Severity Detection.

Class labels follow the APTOS 2019 / ICDR severity coding:
    0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR.

This module is the single source of truth for class names, ICDR staging
criteria, and per-stage clinical guidance shown to the end user.
"""
from typing import Dict, Any, List

# Canonical class keys in label order 0..4
CLASSES: List[str] = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR",
]

# Human-friendly full names
CLASS_DISPLAY_NAMES: Dict[int, str] = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}

# Short display labels used in tables / badges
CLASS_SHORT_NAMES: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

# Per-stage clinical metadata (ICDR-aligned)
SEVERITY_METADATA: Dict[int, Dict[str, Any]] = {
    0: {
        "id": 0,
        "key": "No_DR",
        "name": "No Diabetic Retinopathy",
        "stage": "Stage 0",
        "risk_level": "None / Healthy",
        "badge_color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "icdr_criteria": "No retinal vascular abnormalities detected. Optic disc and fovea appear normal.",
        "clinical_findings": [
            "Normal fundus appearance",
            "No microaneurysms, hemorrhages, or exudates",
            "Clear macula and sharp optic disc margins",
        ],
        "recommended_action": "Continue routine diabetic screening (12-month interval). Maintain systemic glucose and blood-pressure control.",
        "urgency": "Low (12-month routine follow-up)",
        "laser_required": False,
    },
    1: {
        "id": 1,
        "key": "Mild",
        "name": "Mild Non-Proliferative DR",
        "stage": "Stage 1",
        "risk_level": "Mild Risk",
        "badge_color": "#eab308",
        "badge_bg": "rgba(234, 179, 8, 0.15)",
        "icdr_criteria": "Microaneurysms only. Earliest clinically observable phase of diabetic retinopathy.",
        "clinical_findings": [
            "Focal isolated microaneurysms present",
            "No hard exudates or cotton-wool spots",
            "Macular architecture preserved without gross edema",
        ],
        "recommended_action": "Repeat comprehensive dilated fundus examination in 6-12 months. Optimise glycaemic control (HbA1c < 7.0%).",
        "urgency": "Moderate (6-12 month follow-up)",
        "laser_required": False,
    },
    2: {
        "id": 2,
        "key": "Moderate",
        "name": "Moderate Non-Proliferative DR",
        "stage": "Stage 2",
        "risk_level": "Moderate Risk",
        "badge_color": "#f97316",
        "badge_bg": "rgba(249, 115, 22, 0.15)",
        "icdr_criteria": "More than just microaneurysms, but less than severe NPDR criteria.",
        "clinical_findings": [
            "Multiple intraretinal micro-hemorrhages",
            "Hard exudates (lipid leakage)",
            "Cotton-wool spots (nerve-fiber layer infarction)",
        ],
        "recommended_action": "Refer to an ophthalmologist within 3-6 months. Consider Optical Coherence Tomography (OCT) to rule out Diabetic Macular Edema (DME).",
        "urgency": "Elevated (3-6 month specialist review)",
        "laser_required": False,
    },
    3: {
        "id": 3,
        "key": "Severe",
        "name": "Severe Non-Proliferative DR",
        "stage": "Stage 3",
        "risk_level": "High Risk (Pre-Proliferative)",
        "badge_color": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "icdr_criteria": "Meets the 4-2-1 rule: severe intraretinal hemorrhages (>20 in each of 4 quadrants), venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant.",
        "clinical_findings": [
            "Extensive 4-quadrant blot hemorrhages",
            "Venous beading / dilation in multiple quadrants",
            "Intraretinal microvascular abnormalities (IRMA)",
            "High 1-year progression risk to proliferative stage",
        ],
        "recommended_action": "Prompt retinal-specialist consultation within 2-4 weeks. High risk of rapid neovascularization.",
        "urgency": "High (2-4 week urgent specialist review)",
        "laser_required": True,
    },
    4: {
        "id": 4,
        "key": "Proliferate_DR",
        "name": "Proliferative Diabetic Retinopathy",
        "stage": "Stage 4",
        "risk_level": "Critical / Sight-Threatening",
        "badge_color": "#a855f7",
        "badge_bg": "rgba(168, 85, 247, 0.15)",
        "icdr_criteria": "Neovascularization of the disc (NVD) or elsewhere (NVE), preretinal/vitreous hemorrhage, or fibrovascular proliferation.",
        "clinical_findings": [
            "Pathological neovascularization (NVD / NVE)",
            "Preretinal or vitreous hemorrhage",
            "Fibrous proliferation with tractional detachment risk",
            "Severe visual impairment hazard if untreated",
        ],
        "recommended_action": "Urgent retinal intervention within 24-48 hours. Consider Panretinal Photocoagulation (PRP) laser and/or intravitreal anti-VEGF therapy.",
        "urgency": "Critical / Immediate (24-48 hours)",
        "laser_required": True,
    },
}
