# KSL_DL2026: Korean Sign Language Recognition

**COSE474 Deep Learning · Korea University · Spring 2026**

**Group 11:** Hakeemi · Nico · 고동우

---

## Project Summary

Korean Sign Language (KSL) recognition is a critical accessibility problem — approximately 450 million people worldwide are deaf or hard of hearing, and KSL is distinct from other sign languages including ASL. Despite its importance, KSL is severely understudied in the deep learning literature, with very few publicly available datasets.

This project addresses **isolated KSL word recognition** on the KSL-77 dataset using a systematic ablation study. We start from a naive CNN baseline that treats sign recognition as single-frame image classification, demonstrate its fundamental limitation (it cannot capture temporal motion), and progressively improve it using temporal modeling (CNN+LSTM), data augmentation, and transfer learning.

**Core argument:** KSL signs are movements, not static poses. A single-frame CNN achieves only 6.98% accuracy — barely above random chance (1.3%) — because two different signs can look identical in one frozen frame but differ entirely in how the hand moves over time. This motivates the LRCN (Long-term Recurrent Convolutional Network) architecture, which uses CNN to extract per-frame spatial features and LSTM to model the temporal sequence.

**Reference benchmark:** Shin et al. (2023) CNN+Transformer hybrid — 89.00% on KSL-77.

---

## Dataset — KSL-77

| Property | Value |
|---|---|
| Source | Yang et al. (2020), MMM 2020 |
| GitHub | https://github.com/Yangseung/KSL |
| Total videos | 1,540 (1,229 successfully extracted) |
| Sign classes | 77 Korean daily-use words |
| Signers | 20 deaf signers |
| Recording locations | 17 different locations |
| Video length | ~4 seconds at 30fps (~120 frames) |
| Frames sampled | 16 evenly-spaced per clip |
| Frame resolution | 224 × 224 RGB |

**Train/val split:** By signer identity, not randomly by video. Signers 00–15 → train (~984 clips), signers 16–19 → val (~245 clips). This prevents data leakage — if the same signer's videos appeared in both sets, the model could learn to recognize the person rather than the sign.

**Note:** Raw video files and extracted frames are not included in this repository due to size constraints. See notebook `01_data_pipeline.ipynb` for the extraction pipeline.

---

## Repository Structure

```
20261R0136COSE47400/
│
├── README.md
├── config.py
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_baseline_CNN.ipynb
│   ├── 03_CNN_LSTM.ipynb
│   ├── 04_augmentation.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_combined_best.ipynb
│   └── 07_evaluation.ipynb
│
└── results/
    └── logs/
        ├── 02_baseline_cnn_log.csv
        ├── 04_augmentation_log.csv
        └── 04_augmentation_v2_log.csv
```

Run notebooks in order. Each notebook depends on outputs from the previous — `01` extracts frames, `02` to `06` each train a model and save a checkpoint, and `07` loads all checkpoints to generate final evaluation figures.

---

## How to Run

All notebooks run on **Elice AI Cloud** (GPU instance) or **Google Colab** with GPU runtime.

**Setup on Elice:**

```python
import os, sys
sys.path.append('/home/elicer')
import config

config.DATA_FRAMES  = '/home/elicer/frames'
config.MODELS_CKPT  = '/home/elicer/models/checkpoints'
config.RESULTS_LOGS = '/home/elicer/results/logs'
config.RESULTS_FIGS = '/home/elicer/results/figures'
```

**Setup on Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/KSL_DL2026')
import config
```

Switch to GPU runtime before training: Runtime → Change runtime type → GPU.

**Dependencies:** All standard — no custom installs beyond default environment.

```
torch torchvision numpy matplotlib seaborn scikit-learn Pillow csv
```

---

## Configuration — config.py

All shared paths and hyperparameters are centralized in `config.py`. Every notebook imports this file so settings are consistent across all experiments. LRCN notebooks (03–06) override `BATCH_SIZE → 8` and `LEARNING_RATE → 1e-4` locally due to memory constraints from 16-frame sequences.

| Parameter | Value | Description |
|---|---|---|
| `NUM_CLASSES` | 77 | KSL sign classes |
| `NUM_FRAMES` | 16 | frames sampled per video |
| `IMG_SIZE` | 224 | frame resolution (px) |
| `BATCH_SIZE` | 32 | default batch (overridden to 8 in LRCN notebooks) |
| `LEARNING_RATE` | 0.001 | default LR (overridden to 1e-4 in LRCN notebooks) |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `NUM_EPOCHS` | 50 | max training epochs |
| `PATIENCE` | 10 | early stopping patience |
| `CNN_BACKBONE` | vgg16 | pretrained feature extractor |
| `LSTM_HIDDEN` | 256 | LSTM hidden state size |
| `RANDOM_SEED` | 42 | reproducibility seed |

---

## Methodology

### Baseline — Simple CNN (Notebook 02)

**Architecture:** VGG16 (pretrained on ImageNet, frozen) + custom FC head → 77 classes

**Input:** Single middle frame per clip (frame index 8 of 16) — no temporal information

**Why this fails:** KSL signs are defined by motion, not static hand shape. Two different signs can share an identical hand configuration at one frozen moment but differ entirely in how the hand moves. A single-frame CNN is structurally incapable of distinguishing these cases.

**Result: 6.98% val accuracy** (31 epochs, early stopping)

---

### Improvement 1 — CNN+LSTM / LRCN (Notebook 03)

**Architecture:** VGG16 (frozen) → Linear(25088→512) + BatchNorm + ReLU + Dropout(0.2) → LSTM(hidden=256) → FC(77)

**Input:** Full 16-frame sequence per clip

**Key design:** All 16 frames processed by VGG16 in one batched call via reshape trick (B×T, C, H, W), then reshaped back to (B, T, 512) before the LSTM. Significantly faster than looping over frames.

**Hyperparameter finding:** Dropout=0.5 → ~1% accuracy (too aggressive). No dropout → 100% train / 8.5% val (overfit). Dropout=0.2 → best balance.

**Result: 8.53% val accuracy** (+1.55pp over CNN baseline)

---

### Improvement 2 — Data Augmentation (Notebook 04)

**Addresses:** Data scarcity (~12 training clips per class after signer split)

**v1 — Aggressive augmentation:**
Spatial: flip (p=0.5), rotation ±15°, color jitter ±0.3, random resized crop. Temporal: frame speed variation (0.8×–1.2×). Result: **5.43%** — worse than baseline. Training curves showed severe overfitting (train 35%, val 5%). Temporal augmentation disrupted the LSTM's temporal signal.

**v2 — Mild augmentation + label smoothing:**
Spatial: flip (p=0.3), rotation ±8°, color jitter ±0.15. No crop, no temporal augmentation. Added label smoothing (0.1), dropout (0.3), CosineAnnealingLR. Result: **7.36%** — improved +1.93pp over v1 but still -1.17pp below LRCN baseline.

**Finding:** Augmentation consistently underperforms the plain LRCN baseline. With only 16 unique signers, augmentation adds sample variety within existing signers but cannot introduce new signer diversity — the fundamental source of generalization failure. This strongly motivates transfer learning as the primary data scarcity strategy.

---

### Improvement 3 — Transfer Learning (Notebook 05)

**Addresses:** Data scarcity + domain mismatch (ImageNet vs sign language)

**Approach:** Pretrain CNN on a larger sign language dataset, then fine-tune on KSL-77. Low-level hand features — finger shapes, wrist angles, hand orientation — are universal across sign languages. A backbone pretrained on sign language data requires only KSL-specific adaptation.

**Result: TBD**

---

### Combined Best Model (Notebook 06)

LRCN + transfer learning + mild augmentation + label smoothing + fine-tuned backbone. Expected to achieve the highest accuracy by combining the best configurations from all previous experiments.

**Result: TBD**

---

## Ablation Table

| Model | Temporal | Augmentation | Transfer | Val Accuracy |
|---|---|---|---|---|
| Simple CNN (nb 02) | ✗ | ✗ | ✗ | **6.98%** |
| LRCN baseline (nb 03) | ✓ | ✗ | ✗ | **8.53%** |
| LRCN + aug v1 (nb 04) | ✓ | ✓ aggressive | ✗ | **5.43%** |
| LRCN + aug v2 (nb 04) | ✓ | ✓ mild | ✗ | **7.36%** |
| LRCN + transfer (nb 05) | ✓ | ✗ | ✓ | TBD |
| LRCN + best combo (nb 06) | ✓ | ✓ | ✓ | TBD |
| Shin et al. 2023 (reference) | ✓ | — | — | **89.00%** |

Each row isolates one variable — this controlled design ensures any accuracy difference can be attributed to the specific technique, not confounding factors.

---

## Key Findings So Far

**Temporal modeling helps:** LRCN achieves +1.55pp over CNN baseline, confirming KSL signs require sequential modeling.

**Augmentation consistently hurts:** Both aggressive (v1: -3.10pp) and mild (v2: -1.17pp) augmentation underperform the plain LRCN baseline. Root cause: signer diversity cannot be synthesized through spatial transforms on a 16-signer dataset.

**Data scarcity is the dominant problem:** With ~12 training clips per class, standard regularization and augmentation techniques are insufficient. Transfer learning remains the primary untested strategy.

---

## Evaluation

**Primary metric:** Top-1 classification accuracy on held-out val signers (16–19)

**Secondary:** Confusion matrix (77×77) — identifies which sign pairs are systematically confused, providing qualitative insight into model failure modes

**Published baselines:** TSN — 79.80% (Yang et al. 2020), CNN+Transformer — 89.00% (Shin et al. 2023)

---

## Timeline

| Date | Milestone |
|---|---|
| Mar 25 | Team formed, KSL topic confirmed, TA approval received |
| Mar 27 | Drive folder structure, config.py, README set up |
| Apr 15 | Notebook 01 complete — 1,229 clips extracted (Nico) |
| Apr 15 | Notebook 02 complete — 6.98% baseline (Hakeemi) |
| May 4–6 | Midterm presentation |
| May 5 | Notebook 03 complete — 8.53% LRCN baseline (Nico) |
| May 6 | Notebook 04 complete — aug v1 5.43%, aug v2 7.36% (Hakeemi) |
| Jun 10–15 | Final presentation |
| Jun 23 | Final report due (4 pages, CVPR format, English) |

---

## Team Contributions

| Member | Role | Notebooks |
|---|---|---|
| Hakeemi | Model lead | 02, 04, 06 |
| Nico | Data lead | 01, 03, 05 |
| 고동우 | Evaluation + Report lead | 07, slides, report |

---

## References

1. Yang, S., Jung, S., Kang, H., & Kim, C. (2020). The Korean Sign Language Dataset for Action Recognition. MMM 2020.
2. Shin, J., Musa Miah, A.S., et al. (2023). Korean Sign Language Recognition Using Transformer-Based Deep Neural Network. Applied Sciences, 13(5), 3029.
3. Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition and Description. CVPR 2015.
4. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
