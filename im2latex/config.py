from pathlib import Path


# =========================
# Paths
# =========================

ROOT_DIR = Path(".")

# fine-tune делаем на старом im2latex dataset
DATASET_DIR = ROOT_DIR / "datasets" / "im2latex"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"

IMAGES_DIR = DATASET_DIR / "images" / "formula_images_processed"

# новая отдельная папка, чтобы не затереть combined checkpoint
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "im2latex_after_combined_finetune"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

LAST_CKPT = CHECKPOINT_DIR / "last.pt"
BEST_CKPT = CHECKPOINT_DIR / "best.pt"
BEST_EMA_CKPT = CHECKPOINT_DIR / "best_ema.pt"

# откуда стартуем fine-tune
PRETRAIN_CKPT = ROOT_DIR / "checkpoints" / "im2latex_combined_v1" / "best_ema.pt"


# =========================
# Image params
# =========================

IMAGE_HEIGHT = 64
MAX_WIDTH = 512


# =========================
# Tokenizer
# =========================

VOCAB_MIN_FREQ = 2
VOCAB_MAX_SIZE = 12000


# =========================
# Model
# =========================

D_MODEL = 384

ENCODER_VARIANT = "small"
ENCODER_PRETRAINED = True


# =========================
# Training
# =========================

BATCH_SIZE = 2

MAX_LEN = 384
FILTER_MAX_FORMULA_CHARS = 350

USE_LENGTH_BUCKETING = True
BUCKET_SIZE = 512

# это 8 эпох fine-tune, а не обучение с нуля
NUM_EPOCHS = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.02

SAVE_EVERY_STEPS = 500

# effective batch = 2 * 2 = 4
GRAD_ACCUM_STEPS = 2


# =========================
# EMA
# =========================

USE_EMA = True
EMA_DECAY = 0.999


# =========================
# Scheduled Sampling
# =========================

# на fine-tune лучше выключить
USE_SCHEDULED_SAMPLING = False
SS_START_EPOCH = 8
SS_MAX_PROB = 0.05
SS_WARMUP_EPOCHS = 4


# =========================
# Decoding
# =========================

BEAM_SIZE = 5
REPEAT_PENALTY = 1.0


# =========================
# Eval
# =========================

EVAL_BATCH_SIZE = 8
MAX_EVAL_SAMPLES = 1000