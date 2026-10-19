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
├── CBP_Multimodal_With_Augmented_Training_And_Explainability.ipynb
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

The original validation and test sets are kept unchanged throughout model selection and final evaluation.
Augmentation and class rebalancing are applied only to the training data. The original image files and dataset CSV files are not modified.

---

## Multimodal Architecture

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

#### Training Augmentation and Rebalancing
A second robustness experiment introduces training only augmentation and abnormal class rebalancing.
Abnormal samples are oversampled during training, while validation and test distributions remain unchanged.

Training images receive controlled transformations such as:
- Small rotations
- Translation and scaling
- Brightness/contrast variation
- Mild blur and spatial perturbation
Repeated abnormal examples also receive additional text regularization to reduce report memorization and encourage visual learning.

---

## Robustness Model Performance

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

## Augmented Training Robustness Model

The augmented/rebalanced model uses the same DenseNet121 + BioClinicalBERT + residual CBP architecture and robustness framework, but introduces training-only class rebalancing and stochastic augmentation.

The validation-selected operating threshold for this model is:

Threshold = 0.405

Validation multimodal performance:

| Condition | Accuracy | Precision | Recall | F1 Score | MCC | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Multimodal | 0.9188 | 0.7643 | 0.9469 | 0.8459 | 0.7999 | 0.9749 |
| Text Only | 0.4500 | 0.2997 | 1.0000 | 0.4612 | 0.2900 | 0.9564 |
| Image Only | 0.7063 | 0.4333 | 0.8053 | 0.5635 | 0.4114 | 0.8443 |

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
| Text Deletion AUC | 0.2807 | 0.2745 | 0.1212 |
| Text Insertion AUC | 0.4486 | 0.2971 | 0.3193 |
| Image Deletion AUC | 0.5625 | 0.1105 | 0.5539 |
| Image Insertion AUC | 0.6743 | 0.1011 | 0.6785 |
| Joint Deletion AUC | 0.2905 | 0.2677 | 0.1352 |
| Joint Insertion AUC | 0.4919 | 0.2879 | 0.3916 |
| Joint Top-20% Confidence Drop | 0.4761 | 0.3453 | 0.6314 |
| Image Modality Importance | 0.1506 | 0.1086 | 0.1647 |
| Text Modality Importance | 0.0830 | 0.1219 | 0.0893 |

Lower deletion AUC indicates stronger deletion faithfulness, while higher insertion AUC indicates stronger insertion faithfulness.
Positive modality importance indicates that removing that modality reduces confidence in the explained class.
