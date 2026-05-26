from pathlib import Path


# =========================
# Paths
# =========================

ROOT_DIR = Path(".")

DATASET_DIR = ROOT_DIR / "datasets" / "im2latex" / "final_full_plus_school"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"

IMAGES_DIR = DATASET_DIR / "images"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "im2latex_full_plus_school_ft"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

LAST_CKPT = CHECKPOINT_DIR / "last.pt"
BEST_CKPT = CHECKPOINT_DIR / "best.pt"
BEST_EMA_CKPT = CHECKPOINT_DIR / "best_ema.pt"

INFER_CKPT = CHECKPOINT_DIR / "current_bot_best.pt"
INFER_TOKENIZER = TOKENIZER_PATH


# =========================
# Image params
# =========================

IMAGE_HEIGHT = 64
MAX_WIDTH = 512


# =========================
# Tokenizer
# =========================

VOCAB_MIN_FREQ = 2
VOCAB_MAX_SIZE = 11000


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

MAX_LEN = 320
FILTER_MAX_FORMULA_CHARS = 350

USE_LENGTH_BUCKETING = True
BUCKET_SIZE = 256

NUM_EPOCHS = 20

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

SAVE_EVERY_STEPS = 500

GRAD_ACCUM_STEPS = 2


# =========================
# EMA
# =========================

USE_EMA = True
EMA_DECAY = 0.999


# =========================
# Scheduled Sampling
# =========================

USE_SCHEDULED_SAMPLING = False
SS_START_EPOCH = 999
SS_MAX_PROB = 0.0
SS_WARMUP_EPOCHS = 1


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