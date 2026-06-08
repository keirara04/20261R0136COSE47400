# KSL-77 Sign Language Recognition
**COSE474 Deep Learning · Korea University · Spring 2026**  
Group 11: Hakeemi · Nico · 고동우

---

We built a Korean Sign Language recognition system on the KSL-77 dataset as part of our deep learning course project. The core question we kept coming back to: *why does a frozen ImageNet model trained on photos struggle so much with signing videos?* The answer we found was signs are movements, not poses and it ended up driving every architectural decision we made.

This repo documents our full ablation, from a naive single-frame CNN baseline all the way to pose keypoint modeling.

---

## Results

| Model | Val Acc | Notes |
|---|---|---|
| Simple CNN (nb02) | 13.57% | single middle frame, no temporal info |
| LRCN baseline (nb03) | 14.34% | 32 frames, VGG16 + LSTM |
| LRCN + Augmentation (nb04) | 12.02% | augmentation hurts — see findings |
| LRCN + WLASL Transfer (nb05) | **23.26%** | pretrain on ASL, fine-tune on KSL |
| Combined best (nb06) | **25.19%** | grid search over TL + aug + hyperparams |
| Pose-LRCN (nb07) | **25.00%** | MediaPipe keypoints replace RGB pixels |
| Pose-LRCN + Aug (nb08) | 16.00% | augmentation hurts here too |
| Shin et al. 2023 (reference) | 89.00% | CNN+Transformer, different eval protocol |

**Honest caveat on the Shin et al. comparison:** they used a random 70/10/20 split; we use a strict signer-based split (signers 16–19 never appear in training). Same dataset, very different problem difficulty. The gap is mostly evaluation protocol, not architecture.

---

## Dataset

KSL-77 from Yang et al. (2020) — 1,229 video clips across 67 classes (10 of the 77 defined classes have no recordings in the released dataset). 20 deaf signers, 17 filming locations, ~4 seconds at 30fps per clip.

We sample 32 evenly-spaced frames per clip at 224×224. Train/val split is by signer: signers 00–15 train (~984 clips), signers 16–19 val (~245 clips). This matters as a random split lets the model recognize the person signing, not the sign itself. Signer split is harder to utilize.

Raw videos and frames are not in this repo (too large). Run `01_data_pipeline.ipynb` to extract them from the original dataset.

---

## Notebooks

```
notebooks/
  01_data_pipeline.ipynb        frame extraction + dataset verification
  02_baseline_CNN.ipynb         single-frame VGG16 baseline (13.57%)
  03_CNN_LSTM.ipynb             LRCN: VGG16 + LSTM over 32 frames (14.34%)
  04_augmentation.ipynb         spatial augmentation ablation (12.02%)
  05_transfer_learning.ipynb    WLASL pretraining → KSL fine-tune (23.26%)
  06_combined_best.ipynb        hyperparameter grid search on best config
  07_pose_lrcn.ipynb            MediaPipe keypoints + LSTM (25.00%)
  08_pose_lrcn_aug.ipynb        pose + augmentation + BiLSTM (16.00%)
```

Run in order. Each notebook saves a checkpoint to `models/checkpoints/` and training curves + confusion matrices to `results/figures/`.

---

## Setup

Tested on Elice AI Cloud (A100 GPU) and Google Colab.

```bash
pip install torch torchvision mediapipe seaborn scikit-learn Pillow
```

**Elice paths (default):**
```python
BASE_DIR   = Path('.')
FRAMES_DIR = BASE_DIR / 'frames'
CKPT_DIR   = BASE_DIR / 'models' / 'checkpoints'
```

**Colab paths:**
```python
from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = Path('/content/drive/MyDrive/KSL_DL2026')
```

Key settings that differ from `config.py` defaults as LRCN notebooks use `BATCH=8` (GPU memory constraint with 32 frames), `LSTM_HIDDEN=64` (anything larger overfits catastrophically on 12 clips/class), and `DROPOUT=0.4`.

---

## What we found

**Temporal modeling helps, but less than expected.** LRCN gives +0.77pp over CNN. The improvement is real as the model is genuinely learning motion patterns, but data scarcity suppresses it. With ~12 training clips per class, the LSTM ends up memorizing how the 16 training signers move rather than learning the sign geometry.

**Augmentation consistently hurts.** This surprised us. Every augmentation configuration we tried such as aggressive (nb04v1), mild (nb04v2), keypoint-space (nb08), underperformed the non-augmented equivalent. Our interpretation: with only 16 training signers, augmentation creates more variety *within* those 16 people's styles, but the generalization bottleneck is *between* signers. You can't augment your way to new people.

**Transfer learning from ASL actually works.** Pretraining LSTM on WLASL (100-class ASL dataset) then fine-tuning on KSL gave our best pixel-based result at 23.26%, a 9pp jump over the LRCN baseline. The pretrained LSTM has seen ~750 ASL signing sequences and learned something useful about hand motion patterns that transfers across sign languages, even though ASL and KSL aren't mutually intelligible.

**Pose keypoints are the right input modality.** Switching from RGB pixels to MediaPipe skeleton keypoints (53 landmarks × 3D coordinates = 159 features/frame) gives our overall best result at 25%. The model no longer needs to figure out what's a hand and what's a background as that's already done. This also explains why published results are so much higher: most competitive KSL systems use keypoints or depth maps, not raw video.

**Data scarcity is the real problem.** Every experiment points here. 12 clips per class, 16 training signers, no technique overcomes this cleanly. The right fix is more data.

---

## Architecture

The LRCN (Long-term Recurrent Convolutional Network, Donahue et al. CVPR 2015) uses a frozen VGG16 to extract a 512-dim feature vector per frame, then feeds the 32-frame sequence into a single-layer LSTM. Final hidden state → classifier.

```
Input: (B, 32, 3, 224, 224)
  → reshape (B×32, 3, 224, 224)
  → VGG16 features + avgpool
  → Linear(25088→512) + BN + ReLU + Dropout
  → reshape (B, 32, 512)
  → LSTM(512→64)
  → final hidden state (B, 64)
  → Linear(64→67)
```

For the pose notebooks (07/08), VGG16 is replaced by MediaPipe Holistic running on the pre-extracted JPEGs. Each frame becomes a 159-dim keypoint vector. Everything from the LSTM onward is identical.

---

## References

Yang, S. et al. (2020). Korean Sign Language Dataset for Action Recognition. MMM 2020.

Shin, J. et al. (2023). Korean Sign Language Recognition Using Transformer-Based Deep Neural Network. *Applied Sciences*, 13(5), 3029.

Donahue, J. et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition and Description. CVPR 2015.

Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.
