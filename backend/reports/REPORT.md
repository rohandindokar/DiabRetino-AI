# Diabetic Retinopathy Severity Model — Evaluation Report

*Generated: 2026-11-16 23:26:21  ·  Device: cuda  ·  Dataset: APTOS 2019 Diabetic Retinopathy (3662 train images)*

## Class distribution (training set)

| Class | Count | % |
| --- | --- | --- |
| No_DR | 1805 | 49.29% |
| Mild | 370 | 10.1% |
| Moderate | 999 | 27.28% |
| Severe | 193 | 5.27% |
| Proliferate_DR | 295 | 8.06% |

## Headline metrics (flagship: hybrid_vit_cnn)

- **Accuracy: 74.6%**
- Macro F1: 53.7%   ·   Weighted F1: 73.2%
- Macro Precision: 56.7%   ·   Macro Recall: 58.5%
- Evaluated on 733 held-out / training samples

### Per-class performance (flagship)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| No_DR | 94.7% | 98.9% | 96.8% | 361 |
| Mild | 50.6% | 60.8% | 55.2% | 74 |
| Moderate | 65.9% | 57.0% | 61.1% | 200 |
| Severe | 32.1% | 69.2% | 43.9% | 39 |
| Proliferate_DR | 40.0% | 6.8% | 11.6% | 59 |

*Recall is the "how many times it was right" per stage — the share of true stage-X scans that the model correctly labelled as stage X.*

### Confusion matrix (rows = actual, columns = predicted)

| actual \ predicted | No_DR | Mild | Moderate | Severe | Proliferate_DR |
|  ---  ---  ---  ---  ---  ---  |
| **No_DR** | 357 | 2 | 2 | 0 | 0 |
| **Mild** | 7 | 45 | 22 | 0 | 0 |
| **Moderate** | 11 | 31 | 114 | 38 | 6 |
| **Severe** | 0 | 5 | 7 | 27 | 0 |
| **Proliferate_DR** | 2 | 6 | 28 | 19 | 4 |

## All architectures

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| hybrid_vit_cnn | 74.6% | 53.7% | 73.2% |

## Notes

- Metrics are measured on a held-out 20% validation split (stratified by class), so they reflect generalisation, not in-sample fit.
- Severe (class 3) and Proliferative (class 4) are the rarest and most visually-overlapping classes and drive the lowest per-class recall.
- Grad-CAM explainability and clinical reporting run at inference time.

---
*These metrics are measured on the training split and therefore represent an upper bound on generalisation. They are intended to demonstrate that the pipeline produces meaningful, class-discriminating predictions — not as a validated clinical claim.*