# KSL_DL2026: Korean Sign Language Recognition

**COSE474 Deep Learning · Korea University · Spring 2026**

**Group 11:** Hakeemi ·  Nico  · 고동우

---

## Project Summary

Korean Sign Language (KSL) recognition is a critical accessibility problem, approximately 450 million people worldwide are deaf or hard of hearing, and KSL is distinct from other sign languages including ASL. Despite its importance, KSL is severely understudied in the deep learning literature, with very few publicly available datasets.

This project addresses **isolated KSL word recognition** on the KSL-77 dataset using a systematic ablation study. We start from a naive CNN baseline that treats sign recognition as single-frame image classification, demonstrate its fundamental limitation (it cannot capture temporal motion), and progressively improve it using temporal modeling (CNN+LSTM), data augmentation, and transfer learning.

**Core argument:** KSL signs are movements, not static poses. A single-frame CNN achieves only 6.98% accuracy, barely above random chance (1.3%), because two different signs can look identical in one frozen frame but differ entirely in how the hand moves over time. This motivates the LRCN (Long-term Recurrent Convolutional Network) architecture, which uses CNN to extract per-frame spatial features and LSTM to model the temporal sequence.

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

**Train/val split:** By signer identity, not randomly by video. Signers 00–15 → train (~984 clips), signers 16–19 → val (~245 clips). This prevents data leakage: if the same signer's videos appeared in both sets, the model could learn to recognize the person rather than the sign.

**Note:** Raw video files and extracted frames are not included in this repository due to size constraints. See notebook `01_data_pipeline.ipynb` for the extraction pipeline.

---

## Repository Structure

```
20261R0136COSE47400/
│
├── README.md                        ← this file
├── config.py                        ← shared paths + hyperparameters
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb       ← Nico: frame extraction + KSLDataset class
│   ├── 02_baseline_CNN.ipynb        ← Hakeemi: simple CNN baseline (single frame)
│   ├── 03_CNN_LSTM.ipynb            ← Nico: LRCN baseline (CNN + LSTM)
│   ├── 04_augmentation.ipynb        ← Hakeemi: LRCN + data augmentation
│   ├── 05_transfer_learning.ipynb   ← Nico: LRCN + transfer learning
│   ├── 06_combined_best.ipynb       ← Hakeemi: LRCN + aug + transfer (best model)
│   └── 07_evaluation.ipynb          ← 고동우: confusion matrix + ablation table
│
└── results/
    └── logs/
        └── 02_baseline_cnn_log.csv  ← baseline experiment results
```

**Run notebooks in order.** Each notebook depends on outputs from the previous:
- `01` extracts frames → `data/frames/`
- `02–06` each train a model and save to `models/checkpoints/`
- `07` loads all checkpoints and generates final evaluation figures

---

## How to Run

### Prerequisites

All notebooks run in **Google Colab** with GPU runtime. The project uses Google Drive for shared storage.

### Setup

1. Mount Google Drive in every notebook (first cell):
```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/KSL_DL2026')
import config
```

2. Switch to GPU runtime before training:
   `Runtime → Change runtime type → GPU (T4 or A100)`

### Dependencies

All standard: no custom installs needed beyond default Colab:
```
torch torchvision numpy matplotlib seaborn sklearn PIL csv
```

---

## Configuration — config.py

All shared paths and hyperparameters are centralized in `config.py`. Every notebook imports this file so settings are consistent across all experiments.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `NUM_CLASSES` | 77 | KSL sign classes |
| `NUM_FRAMES` | 16 | frames sampled per video |
| `IMG_SIZE` | 224 | frame resolution (px) |
| `BATCH_SIZE` | 32 | training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `DROPOUT_RATE` | 0.5 | dropout before classifier |
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

**Why this fails:** KSL signs differ in motion, not just hand shape. A frozen frame cannot distinguish signs with identical hand positions but different movement trajectories.

**Result: 6.98% val accuracy** (random chance = 1.3%, 31 epochs, early stopping)

---

### Improvement 1 — CNN + LSTM / LRCN (Notebook 03)

**Architecture:** VGG16 feature extractor → LSTM (hidden=256) → FC(77)

**Input:** Full 16-frame sequence per clip — temporal information included

**Why this helps:** CNN extracts per-frame spatial features → 16 feature vectors → LSTM reads the sequence and models how the hand moves over time → final hidden state predicts the sign.

**Expected improvement:** Published LRCN baseline achieves ~65–75% on KSL-77.

---

### Improvement 2 — Data Augmentation (Notebook 04)

**Addresses:** Data scarcity (~16 training videos per class)

**Spatial augmentation (per frame):**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness ±0.3, contrast ±0.3)
- Random resized crop (scale 0.9–1.0)

**Temporal augmentation (per clip):**
- Frame speed variation (factor 0.8×–1.2×) — simulates different signing speeds by adjusting the frame sampling window

**Val transforms:** Always clean resize + normalize only — augmentation is never applied to val.

---

### Improvement 3 — Transfer Learning (Notebook 05)

**Addresses:** Data scarcity + domain mismatch between ImageNet and KSL

**Approach:** Pretrain CNN on a larger sign language dataset (ASL), then fine-tune on KSL-77.

**Why ASL→KSL transfer works:** Low-level hand features — finger shapes, wrist angles, hand orientation — are universal across sign languages. The pretrained backbone already understands hands; it only needs to learn KSL-specific motion patterns.

---

### Combined Best Model (Notebook 06)

LRCN + augmentation + transfer learning together. Expected to achieve the highest accuracy by addressing all three identified limitations simultaneously.

---

## Ablation Table

| Model | Temporal (LSTM) | Augmentation | Transfer Learning | Val Accuracy |
|---|---|---|---|---|
| Simple CNN (ours) | ✗ | ✗ | ✗ | **6.98%** |
| LRCN baseline | ✓ | ✗ | ✗ | TBD |
| LRCN + augmentation | ✓ | ✓ | ✗ | TBD |
| LRCN + transfer learning | ✓ | ✗ | ✓ | TBD |
| LRCN + aug + transfer (best) | ✓ | ✓ | ✓ | TBD |
| Shin et al. 2023 (reference) | ✓ | — | — | **89.00%** |

Each row isolates one variable — this controlled design ensures any accuracy difference can be attributed to the specific technique, not confounding factors.

---

## Evaluation

**Primary metric:** Top-1 classification accuracy on held-out val signers

**Secondary:** Confusion matrix (77×77) — identifies which sign pairs are systematically confused and why (e.g. signs with similar hand shapes but different motion)

**Baselines for comparison:**
- TSN = 79.80% (Yang et al. 2020)
- CNN+Transformer (Shin et al. 2023) = 89.00%

---

## Timeline

| Date | Milestone |
|---|---|
| Mar 25 | Team formed, topic confirmed, TA approval received |
| Mar 27 | Drive folder structure, config.py, README set up |
| Apr 15 | Notebook 01 complete — 1,229 clips extracted (Nico) |
| Apr 15 | Notebook 02 complete — 6.98% baseline result (Hakeemi) |
| May 4–6 | Midterm presentation |
| Jun 10–15 | Final presentation |
| Jun 23 | Final report due (4 pages, CVPR format, English) |

---

## Team Contributions

| Member | Role | Notebooks |
|---|---|---|
|  Hakeemi | Model lead | 02, 04, 06 |
|  Nico  | Data lead | 01, 03, 05 |
| 고동우 | Evaluation + Report lead | 07, slides, report |

---

## References

1. Yang, S., Jung, S., Kang, H., & Kim, C. (2020). The Korean Sign Language Dataset for Action Recognition. *MMM 2020*.
2. Shin, J., Musa Miah, A.S., et al. (2023). Korean Sign Language Recognition Using Transformer-Based Deep Neural Network. *Applied Sciences, 13*(5), 3029.
3. Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition and Description. *CVPR 2015*.
4. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*.
