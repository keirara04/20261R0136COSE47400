# KSL_DL2026: Korean Sign Language Recognition

**COSE474 Deep Learning · Korea University · Spring 2026**

**Group 11:** Hakeemi · Nico · 고동우

---

## Project Summary

Korean Sign Language (KSL) recognition is a critical accessibility problem — approximately 450 million people worldwide are deaf or hard of hearing, and KSL is distinct from other sign languages including ASL. Despite its importance, KSL is severely understudied in the deep learning literature, with very few publicly available datasets.

This project addresses isolated KSL word recognition on the KSL-77 dataset through a systematic ablation study. We start from a naive CNN baseline that treats sign recognition as single-frame image classification, demonstrate its fundamental limitation (it cannot capture temporal motion), and progressively investigate temporal modeling (CNN+LSTM), data augmentation, and transfer learning.

**Core argument:** KSL signs are movements, not static poses. A single-frame CNN achieves only 13.57% accuracy because two different signs can look identical in one frozen frame but differ entirely in how the hand moves over time. This motivates the LRCN architecture, which uses CNN to extract per-frame spatial features and LSTM to model the temporal sequence across 32 frames.

**Reference benchmark:** Shin et al. (2023) CNN+Transformer — 89.00% on KSL-77.

---

## Dataset — KSL-77

| Property | Value |
|---|---|
| Source | Yang et al. (2020), MMM 2020 |
| GitHub | https://github.com/Yangseung/KSL |
| Total videos | 1,540 (1,229 successfully extracted) |
| Sign classes | 77 defined (67 present in dataset) |
| Missing classes | 10 classes absent from dataset |
| Signers | 20 deaf signers |
| Recording locations | 17 different locations |
| Video length | ~4 seconds at 30fps |
| Frames sampled | 32 evenly-spaced per clip |
| Frame resolution | 224 × 224 RGB |

**Missing classes (10):** read, next, be friendly, number, guide, parents, 10 minutes, education, visit, far

**Train/val split:** By signer identity — signers 00–15 train (~984 clips), signers 16–19 val (~245 clips). This prevents data leakage. Random split allows the same signer's appearance to leak into both sets, causing the model to recognize the person rather than the sign and inflating accuracy artificially. Signer-based split simulates real deployment where the system must generalize to completely new users.

**Note:** Raw videos and extracted frames are not included in this repository due to size. See `01_data_pipeline.ipynb` for the extraction pipeline.

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
        ├── 03_lrcn_log.csv
        └── 04_augmentation_v2_log.csv
```

Run notebooks in order. Notebooks 01 extracts frames, 02–06 each train one model and save a checkpoint, and 07 loads all checkpoints to generate final evaluation figures.

---

## How to Run

All notebooks run on **Elice AI Cloud** (GPU instance) or **Google Colab** with GPU runtime. Notebooks use local paths — no Google Drive mount required on Elice.

**Setup on Elice:**

```python
import os
BASE_DIR   = '/home/elicer'
FRAMES_DIR = f'{BASE_DIR}/frames'
CKPT_DIR   = f'{BASE_DIR}/models/checkpoints'
FIGS_DIR   = f'{BASE_DIR}/results/figures'
LOGS_DIR   = f'{BASE_DIR}/results/logs'
```

**Setup on Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/KSL_DL2026')
import config
```

**Dependencies:**
```
torch torchvision numpy matplotlib seaborn scikit-learn Pillow csv
```

**Important:** LRCN notebooks (03–06) use `BATCH = 8` and `LEARNING_RATE = 1e-4` locally, overriding config defaults. This is a memory constraint — 32 frames per clip at batch 32 would exceed GPU memory.

---

## Configuration

All shared hyperparameters are in `config.py`. Individual notebooks override as needed.

| Parameter | Config Default | LRCN Notebooks | Description |
|---|---|---|---|
| `NUM_CLASSES` | 77 | 67 (dynamic) | dynamic from dataset |
| `NUM_FRAMES` | 16 | 32 | frames sampled per clip |
| `IMG_SIZE` | 224 | 224 | frame resolution (px) |
| `BATCH_SIZE` | 32 | 8 | memory constraint for LRCN |
| `LEARNING_RATE` | 0.001 | 1e-4 | lower for LSTM stability |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 | L2 regularization |
| `NUM_EPOCHS` | 50 | 50 | max training epochs |
| `PATIENCE` | 10 | 10 | early stopping patience |
| `CNN_BACKBONE` | vgg16 | vgg16 | pretrained feature extractor |
| `LSTM_HIDDEN` | 256 | 64 | reduced to prevent overfitting |
| `RANDOM_SEED` | 42 | 42 | reproducibility |

---

## Methodology

### Design Decisions

**Signer-based split (not random):** Prevents data leakage. Random split inflates accuracy by letting the model recognize signer appearance. Our evaluation is stricter but more representative of real-world deployment.

**Dynamic label remapping:** 10 classes missing from KSL-77. We detect present classes from the dataset, remap to contiguous 0..66 labels, and map indices back to KSL word names for confusion matrix analysis.

**Frozen VGG16 backbone:** With only 12 training clips per class, full fine-tuning would catastrophically overfit. ImageNet features (edges, shapes, textures) transfer reasonably to hand recognition. Only frame_fc, LSTM, and classifier are trained.

**Batched frame processing (reshape trick):** Instead of looping through frames, we reshape `(B, T, C, H, W) → (B*T, C, H, W)` to process all frames in one VGG16 call, then reshape back to `(B, T, 512)` for the LSTM. Significantly faster by exploiting GPU parallelism.

**Controlled ablation design:** Each experiment changes exactly one variable. Same split, same val transforms, same backbone, same metric across all experiments. Any accuracy difference is attributable to the specific technique being tested.

**Aggressive regularization:** Hyperparameter search showed that standard settings catastrophically overfit on KSL-77. LSTM hidden size reduced from 256 to 64, dropout increased from 0.2 to 0.4, label smoothing 0.1 added. Model capacity must be proportional to dataset size.

---

### Baseline CNN (Notebook 02)

**Architecture:** VGG16 (frozen, ImageNet) + FC head → 67 classes

**Input:** Single middle frame per clip (frame index 16 of 32) — no temporal information

**Training:** Adam lr=0.001, CrossEntropyLoss, batch=64, early stopping patience=10

**Result: 13.57% val accuracy** (50 epochs, random chance = 1.5%)

**Key observations:**
- 9× above random chance — model learned meaningful spatial features
- Training curves show overfitting — train accuracy climbs while val plateaus early
- Confusion matrix shows systematic confusion between signs sharing similar hand shapes
- Proves single-frame classification is structurally insufficient for KSL — temporal modeling is necessary

---

### LRCN — CNN+LSTM (Notebook 03)

**Architecture:** VGG16 (frozen) → Linear(25088→512) + BatchNorm + ReLU + Dropout(0.4) → LSTM(hidden=64) → FC(67)

**Input:** Full 32-frame sequence per clip

**Forward pass:** Reshape (B,T,C,H,W) → (B×T,C,H,W) → VGG16 → (B×T,512) → reshape (B,T,512) → LSTM → final hidden state → classifier

**Hyperparameter search:**

| Frames | LSTM Hidden | Dropout | Label Smooth | Val Acc |
|---|---|---|---|---|
| 48 | 256 | 0.2 | ✗ | 2.71% |
| 32 | 128 | 0.2 | ✗ | 13.18% |
| 32 | 64 | 0.4 | ✓ | **14.34%** |

**Result: 14.34% val accuracy** (44 epochs, +0.77pp over CNN baseline)

**Key observations:**
- Temporal modeling provides measurable improvement over single-frame CNN
- Improvement is modest — data scarcity suppresses the full benefit of LSTM
- 48 frames caused catastrophic overfitting (train 54%, val 2.71%) — LSTM memorized training signer movement patterns
- Model capacity must be aggressively constrained for datasets this small
- Val accuracy noisy throughout — jumping between 9–14% reflects limited val set size (245 clips)

---

### Data Augmentation (Notebook 04)

**Hypothesis:** Spatial augmentation would increase effective training set size and improve generalization.

**v1 — Aggressive:** flip p=0.5, rotation ±15°, color jitter ±0.3, random crop, temporal speed variation (0.8×–1.2×). Result: **5.43%** — severe overfitting (train 35%, val 5%).

**v2 — Mild (current):** flip p=0.3, rotation ±8°, color jitter ±0.15, no crop, no temporal aug, label smoothing 0.1, dropout 0.4, CosineAnnealingLR. Result: **12.02%** — still -2.32pp below LRCN baseline.

**Augmentation results summary:**

| Version | Val Acc | vs LRCN |
|---|---|---|
| v1 aggressive | 5.43% | -8.91pp |
| v2 mild (old 16-frame) | 7.36% | — |
| v2 mild (current 32-frame) | 12.02% | -2.32pp |

**Key finding:** Augmentation consistently underperforms plain LRCN across all configurations. Root cause: with only 16 unique training signers, augmentation adds sample variety within existing signers but cannot introduce new signer diversity — the fundamental source of generalization failure. This robust negative result across three configurations confirms that signer diversity, not sample variety, is the binding constraint.

---

### Transfer Learning (Notebook 05)

**Addresses:** Signer diversity constraint + domain mismatch (ImageNet vs sign language)

**Approach:** Pretrain CNN on a larger sign language dataset, then fine-tune on KSL-77. Low-level hand features — finger shapes, wrist angles, hand orientation — are universal across sign languages.

**Result: TBD**

---

### Combined Best Model (Notebook 06)

LRCN + transfer learning + best regularization. Expected to achieve highest accuracy.

**Result: TBD**

---

## Ablation Table

| Model | Temporal | Augmentation | Transfer | Val Accuracy |
|---|---|---|---|---|
| Simple CNN (nb 02) | ✗ | ✗ | ✗ | **13.57%** |
| LRCN baseline (nb 03) | ✓ LSTM | ✗ | ✗ | **14.34%** |
| LRCN + aug v2 (nb 04) | ✓ LSTM | ✓ mild | ✗ | **12.02%** |
| LRCN + transfer (nb 05) | ✓ LSTM | ✗ | ✓ | TBD |
| LRCN + best combo (nb 06) | ✓ LSTM | ✓ | ✓ | TBD |
| Shin et al. 2023 (reference) | ✓ | — | — | **89.00%** |

---

## Key Findings

**Temporal modeling helps but modestly:** LRCN achieves +0.77pp over CNN baseline. The improvement is real but suppressed by data scarcity — the LSTM still overfits to training signers' movement styles.

**Augmentation consistently hurts:** All augmentation variants underperform plain LRCN. Signer diversity cannot be synthesized through spatial or temporal transforms on a 16-signer training set.

**Model capacity must match dataset size:** LRCN with hidden=256 and 48 frames catastrophically overfits (2.71%). Reducing to hidden=64, 32 frames, dropout=0.4 achieves the best result (14.34%). Aggressive regularization is essential for datasets this small.

**Data scarcity is the dominant constraint:** Every experiment points to the same root cause — 12 training clips per class across 16 signers is insufficient for any standard deep learning technique to generalize well. Transfer learning is the primary remaining strategy to address this.

---

## Limitations

**Data scarcity:** ~12 training clips per class after signer split. Root cause of overfitting across all experiments. Standard deep learning techniques require orders of magnitude more data per class.

**Signer diversity:** Only 16 unique training signers. Even with thousands of augmented clips, the model has only seen 16 people's signing styles. Transfer learning from larger sign datasets is the primary mitigation strategy.

**Evaluation strictness:** Signer-based split is significantly harder than the random split used in most published KSL results. The gap between our numbers and published results (~14% vs ~65–75% for LRCN) is largely explained by evaluation protocol, not architecture quality.

**Missing classes:** 10 of 77 KSL classes absent from dataset. A deployed system would fail on these signs entirely.

**Single modality:** RGB frames only — no depth, no skeleton keypoints, no hand segmentation. Background and clothing variation across 17 locations adds noise the model must learn to ignore.

---

## Evaluation

**Primary metric:** Top-1 val accuracy on held-out signers 16–19

**Secondary:** Confusion matrix with KSL word labels — identifies which sign pairs are systematically confused and why

**Published baselines:** TSN 79.80% (Yang et al. 2020), CNN+Transformer 89.00% (Shin et al. 2023)

---

## Timeline

| Date | Milestone |
|---|---|
| Mar 25 | Team formed, topic confirmed, TA approval |
| Mar 27 | Drive folder, config.py, README, group chat set up |
| Apr 15 | Notebook 01 complete — 1,229 clips extracted (Nico) |
| Apr 15 | Notebook 02 complete — 13.57% baseline (Hakeemi) |
| May 4–6 | Midterm presentation |
| May 5 | Notebook 03 complete — 14.34% LRCN (Hakeemi) |
| May 6 | Notebook 04 complete — 12.02% aug v2 (Hakeemi) |
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
