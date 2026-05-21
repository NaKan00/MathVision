from pathlib import Path


# =========================
# Paths
# =========================

ROOT_DIR = Path(".")

DATASET_DIR = ROOT_DIR / "datasets" / "im2latex"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"

IMAGES_DIR = DATASET_DIR / "images" / "formula_images_processed"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "im2latex_convnext"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

LAST_CKPT = CHECKPOINT_DIR / "last.pt"
BEST_CKPT = CHECKPOINT_DIR / "best.pt"


# =========================
# Image params
# =========================

IMAGE_HEIGHT = 64
MAX_WIDTH = 512


# =========================
# Tokenizer
# =========================

VOCAB_MIN_FREQ = 2
VOCAB_MAX_SIZE = 8000


# =========================
# Model
# =========================

D_MODEL = 256

ENCODER_VARIANT = "small"
ENCODER_PRETRAINED = True


# =========================
# Training
# =========================

BATCH_SIZE = 4

MAX_LEN = 256
FILTER_MAX_FORMULA_CHARS = 350

USE_LENGTH_BUCKETING = True
BUCKET_SIZE = 512

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

SAVE_EVERY_STEPS = 500

GRAD_ACCUM_STEPS = 1

USE_EMA = True
EMA_DECAY = 0.999
BEST_EMA_CKPT = CHECKPOINT_DIR / "best_ema.pt"


# =========================
# Decoding
# =========================

BEAM_SIZE = 5

REPEAT_PENALTY = 1.05


# =========================
# Eval
# =========================

EVAL_BATCH_SIZE = 8

MAX_EVAL_SAMPLES = 1000