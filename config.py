# ============================================================
# config.py — KSL_DL2026 Project Configuration
# COSE474 Deep Learning, Korea University, Spring 2026
# ============================================================
# USAGE: Add these two lines after mounting Drive in every notebook:
#   import sys
#   sys.path.append('/content/drive/MyDrive/KSL_DL2026')
#   import config
# Then use config.DATA_FRAMES, config.NUM_CLASSES, etc.
# ============================================================

# ── Project root ────────────────────────────────────────────
PROJECT = '/content/drive/MyDrive/KSL_DL2026'

# ── Data paths ──────────────────────────────────────────────
DATA_RAW        = f'{PROJECT}/data/raw'           # original KSL-77 videos — READ ONLY
DATA_FRAMES     = f'{PROJECT}/data/frames'        # extracted frames (output of 01)
DATA_AUG        = f'{PROJECT}/data/augmented'     # augmented frames (output of 04)

# ── Model paths ─────────────────────────────────────────────
MODELS_CKPT     = f'{PROJECT}/models/checkpoints'
MODELS_FINAL    = f'{PROJECT}/models/final'

# Checkpoint filenames — one per experiment
CKPT_BASELINE   = f'{MODELS_CKPT}/baseline_cnn.pth'
CKPT_LRCN       = f'{MODELS_CKPT}/lrcn.pth'
CKPT_AUG        = f'{MODELS_CKPT}/lrcn_augmented.pth'
CKPT_TRANSFER   = f'{MODELS_CKPT}/lrcn_transfer.pth'
CKPT_BEST       = f'{MODELS_FINAL}/best_model.pth'

# ── Results paths ───────────────────────────────────────────
RESULTS_LOGS    = f'{PROJECT}/results/logs'
RESULTS_FIGS    = f'{PROJECT}/results/figures'
ABLATION_TABLE  = f'{PROJECT}/results/ablation_table.csv'

# ── Dataset config ──────────────────────────────────────────
NUM_CLASSES     = 77        # number of KSL sign classes
NUM_SIGNERS     = 20        # total signers in KSL-77
NUM_FOLDS       = 5         # 5-fold cross validation
TOTAL_VIDEOS    = 1540      # total videos in dataset

# ── Frame sampling ──────────────────────────────────────────
NUM_FRAMES      = 16        # frames sampled per video clip
IMG_SIZE        = 224       # resize each frame to 224x224 (ImageNet standard)
FPS_ORIGINAL    = 30        # original video frame rate

# ── Training hyperparameters ────────────────────────────────
BATCH_SIZE      = 32        # training batch size
LEARNING_RATE   = 0.001     # initial learning rate (Adam)
WEIGHT_DECAY    = 1e-4      # L2 regularization
DROPOUT_RATE    = 0.5       # dropout rate before classifier
NUM_EPOCHS      = 50        # max training epochs
PATIENCE        = 10        # early stopping patience (epochs without improvement)

# ── CNN feature extractor ───────────────────────────────────
CNN_BACKBONE    = 'vgg16'   # options: 'vgg16', 'resnet50', 'resnet18'
CNN_FEATURE_DIM = 512       # output feature dimension from CNN (VGG16 default)
FREEZE_CNN      = True      # freeze CNN weights for transfer learning experiment

# ── LSTM config ─────────────────────────────────────────────
LSTM_HIDDEN     = 256       # LSTM hidden state size
LSTM_LAYERS     = 1         # number of LSTM layers
LSTM_BIDIRECT   = False     # bidirectional LSTM (set True to experiment)

# ── Augmentation config ─────────────────────────────────────
AUG_FLIP        = True      # random horizontal flip
AUG_ROTATION    = 15        # max rotation degrees (+/-)
AUG_BRIGHTNESS  = 0.3       # brightness jitter factor
AUG_CONTRAST    = 0.3       # contrast jitter factor
AUG_CROP        = 0.9       # random crop scale (0.9 = crop to 90% then resize back)
AUG_SPEED_MIN   = 0.8       # temporal aug: min frame speed factor
AUG_SPEED_MAX   = 1.2       # temporal aug: max frame speed factor

# ── ImageNet normalization ──────────────────────────────────
# Use these for any pretrained CNN backbone
NORMALIZE_MEAN  = [0.485, 0.456, 0.406]
NORMALIZE_STD   = [0.229, 0.224, 0.225]

# ── Evaluation ──────────────────────────────────────────────
TOPK            = (1, 5)    # compute top-1 and top-5 accuracy

# ── Reproducibility ─────────────────────────────────────────
RANDOM_SEED     = 42        # set this everywhere for reproducible results

# ── Quick sanity check ──────────────────────────────────────
if __name__ == '__main__':
    import os
    print("=" * 50)
    print("KSL_DL2026 — Config loaded")
    print("=" * 50)
    print(f"Project root : {PROJECT}")
    print(f"Classes      : {NUM_CLASSES}")
    print(f"Frames/clip  : {NUM_FRAMES}")
    print(f"Image size   : {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"CNN backbone : {CNN_BACKBONE}")
    print(f"LSTM hidden  : {LSTM_HIDDEN}")
    print()
    print("Checking paths...")
    paths = {
        'data/raw'         : DATA_RAW,
        'data/frames'      : DATA_FRAMES,
        'data/augmented'   : DATA_AUG,
        'models/ckpt'      : MODELS_CKPT,
        'models/final'     : MODELS_FINAL,
        'results/logs'     : RESULTS_LOGS,
        'results/figures'  : RESULTS_FIGS,
    }
    for name, path in paths.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {name}")
