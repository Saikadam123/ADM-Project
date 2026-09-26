# Robust and Explainable Multimodal System for Healthcare under Missing and Noisy Modalities

## Project Overview

The project aims to evaluate and improve the robustness of multimodal healthcare models when one or more modalities are noisy or entirely missing, while keeping computational complexity in mind.

It develops and evaluates a multimodal deep learning system for binary chest X-ray abnormality classification using two complementary sources of information:

- Chest X-ray images
- Associated clinical text (radiology reports)

The prediction task is:

- `0` = No Finding / Normal
- `1` = Any Finding / Abnormal

The project progresses from a baseline multimodal model to an optimized DenseNet121 based model, followed by robustness training and perturbation based explainability.

---

## Main Objectives

The project focuses on four main aspects:

1. Multimodal classification using image and text information.
2. Modality contribution analysis through image only and text only ablation.
3. Robustness to missing or noisy modalities using robust training.
4. Post-hoc explainability using perturbation-based feature and modality analysis.

---

## Repository Structure

```text
ADM-Project/
├── Baseline_Model.ipynb
├── CBP_Multimodal_Baseline_Optimized.ipynb
├── CBP_Multimodal_Robustness_Model_With_Explainability.ipynb
├── IO_Model.ipynb
└── README.md
```

---

## Dataset

The experiments use paired chest X-ray images and radiology reports from the Indiana University Chest X-Ray dataset.

The processed dataset used in the experiments contains:

| Split | Samples |
|---|---:|
| Training | 2,239 |
| Validation | 480 |
| Test | 480 |
| **Total** | **3,199** |

The binary target is defined as:

- `0` = No Finding
- `1` = Any Finding

---

## Final Multimodal Architecture

### Image Branch

The final model uses a CheXpert-pretrained DenseNet121 image encoder.

```text
Chest X-ray
   ↓
DenseNet121
   ↓
Spatial feature map
   ↓
Attention pooling
   ↓
Image embedding
```

### Text Branch

The associated radiology report is encoded using BioClinicalBERT.

```text
Radiology report
   ↓
BioClinicalBERT
   ↓
Token representations
   ↓
Attention pooling
   ↓
Text embedding
```

### Multimodal Fusion

Image and text representations are fused using Residual Compact Bilinear Pooling (CBP) to model cross-modal interactions before binary classification.

```text
Image embedding ─┐
                 ├─ Residual CBP ─ Classifier ─ P(Abnormal)
Text embedding ──┘
```


### Robustness Training

The final model includes several methods to reduce excessive dependence on one modality.

#### 1. Modality Dropout

During training, image or text features may be removed so that the model learns to operate under multimodal and missing modality conditions.

#### 2. Latent Gaussian Noise

Gaussian noise is added to intermediate modality representations during training to improve robustness to latent perturbations.

#### 3. OGM-GE

During training, OGM-GE monitors the learning strength of the image and text modalities and prevents one modality from dominating the other.

#### 4. Missing Modality Substitution

Two missing modality strategies are evaluated:

- Zero substitution
- Calibrated latent substitution

---

## Final Test Performance

The final robust trained model was evaluated on the held-out test set using a validation selected threshold.

| Threshold | Accuracy | Precision | Recall | F1 Score | MCC | AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.475 | 0.9146 | 0.7647 | 0.9204 | 0.8353 | 0.7844 | 0.9653 |

### Missing-Modality Robustness

The final model was also evaluated when one modality was unavailable. Two substitution strategies were considered: zero substitution and calibrated latent substitution.

| Substitution | Condition | Accuracy | Precision | Recall | F1 Score | MCC | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero | Multimodal | 0.9146 | 0.7647 | 0.9204 | 0.8353 | 0.7844 | 0.9653 |
| Zero | Text Only | 0.7333 | 0.4684 | 0.9823 | 0.6343 | 0.5422 | 0.9601 |
| Zero | Image Only | 0.7583 | 0.4909 | 0.7168 | 0.5827 | 0.4358 | 0.8040 |
| Calibrated | Multimodal | 0.9146 | 0.7647 | 0.9204 | 0.8353 | 0.7844 | 0.9653 |
| Calibrated | Text Only | 0.9146 | 0.8214 | 0.8142 | 0.8178 | 0.7620 | 0.9649 |
| Calibrated | Image Only | 0.8146 | 0.6250 | 0.5310 | 0.5742 | 0.4591 | 0.8051 |

Calibrated substitution substantially improves text-only threshold based performance and increases image-only accuracy and precision, while AUROC remains similar to the corresponding zero-substitution setting.

---

## Explainability

The final robustness model is analyzed using a perturbation based multimodal explainability approach.

The method includes:

- Whole word occlusion for text
- Patch occlusion for chest X-ray images
- Whole modality ablation
- Deletion and insertion faithfulness evaluation

---

### Explainability Results

The explainability evaluation was performed on the complete 480 sample held-out test set.

| Metric | Mean | SD | Median |
|---|---:|---:|---:|
| Text Deletion AUC | 0.6008 | 0.1147 | 0.6032 |
| Text Insertion AUC | 0.6471 | 0.1009 | 0.6550 |
| Image Deletion AUC | 0.6215 | 0.0787 | 0.6306 |
| Image Insertion AUC | 0.6686 | 0.0833 | 0.6743 |
| Joint Deletion AUC | 0.5330 | 0.0837 | 0.5427 |
| Joint Insertion AUC | 0.6197 | 0.0762 | 0.6234 |
| Joint Top-20% Confidence Drop | 0.1903 | 0.0996 | 0.1797 |
| Image Modality Importance | 0.1916 | 0.0998 | 0.1955 |
| Text Modality Importance | 0.0848 | 0.1076 | 0.0748 |

Lower deletion AUC indicates stronger deletion faithfulness, while higher insertion AUC indicates stronger insertion faithfulness.
Positive modality importance indicates that removing that modality reduces confidence in the explained class.
