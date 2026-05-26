# MathVision — Image-to-LaTeX Recognition System

MathVision is a neural network system for recognizing mathematical expressions from images and converting them into LaTeX code.

The project combines:

- ConvNeXt visual encoder
- Transformer autoregressive decoder
- symbolic LaTeX tokenization
- beam search decoding
- LaTeX postprocessing
- Telegram bot interface
- dataset preparation and cleaning pipeline
- synthetic dataset generation for short school-level formulas

The system receives an image of a mathematical expression and returns the recognized LaTeX expression as text.

---

# Features

- Recognition of printed mathematical formulas
- Recognition of handwritten mathematical expressions
- Image-to-LaTeX neural recognition pipeline
- ConvNeXt-small encoder + Transformer decoder architecture
- Beam search decoding
- LaTeX tokenization and postprocessing
- Telegram bot interface
- Dataset conversion and cleaning scripts
- Synthetic formula dataset generation
- Evaluation with Char-BLEU, Exact Match, Canonical Exact Match, and Edit Similarity

---

# Project Structure

```text
MathVision/
├── im2latex/                            # neural recognition model
│   ├── config.py
│   ├── train.py
│   ├── infer.py
│   ├── bleu_eval.py
│   ├── canonical.py
│   └── postprocess.py
│
├── tgBot/                               # Telegram bot
│   ├── bot.py
│   └── images/
│
├── scripts/
│   ├── generate_school_symbols.py
│   ├── render_school_symbols.py
│   ├── split_school_symbols.py
│   ├── make_final_full_plus_school.py
│   └── clean_missing_images.py
│
├── datasets/
├── checkpoints/
├── test_images/
│
├── run_train.py
├── run_eval.py
├── run_infer.py
├── analyze_errors.py
├── requirements.txt
└── README.md
```

---

# 1. Clone Repository

```bash
git clone https://github.com/NaKan00/MathVision.git
cd MathVision
```

---

# 2. Create Virtual Environment

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Dataset Setup

Large datasets and checkpoints are not stored in GitHub.

Main datasets used in the project:

- HME100K
- MathWriting
- CROHME
- im2latex100k

Base im2latex dataset:

```text
https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k
```

Expected dataset structure:

```text
MathVision/
└── datasets/
    └── im2latex/
        ├── train.csv
        ├── val.csv
        └── images/
```

---

# 5. Synthetic Dataset Pipeline

The project includes a synthetic dataset generation pipeline for short school-level formulas and visually similar mathematical symbols.

Synthetic dataset name:

```text
school_symbols
```

Generate formulas:

```bash
python scripts/generate_school_symbols.py
```

Render formulas into images:

```bash
python scripts/render_school_symbols.py
```

Split train / validation:

```bash
python scripts/split_school_symbols.py
```

Synthetic dataset statistics:

```text
generated formulas:           ~50,000
successfully rendered images: ~49,780
failed renders:               ~220
train samples:                47,291
validation samples:            2,489
```

Example formulas:

```latex
x^2+y^2=z^2
E=mc^2
\frac{v^2}{c^2}
\sqrt{x^2+y^2}
\cos(x)+\sin(y)=0
\int_0^1 x^2 dx
```

---

# 6. Build Final Unified Dataset

Create the final mixed dataset:

```bash
python scripts/make_final_full_plus_school.py
```

Final dataset:

```text
datasets/im2latex/final_full_plus_school/
```

Dataset composition:

- HME100K
- MathWriting
- CROHME
- original im2latex
- simple printed formulas
- synthetic school-symbol formulas

Final dataset size:

```text
train: 446,256 samples
val:   40,559 samples
```

---

# 7. Quick Installation Check

```bash
python -m im2latex.quick_check
```

Expected result:
the script should load a small batch of images and formulas without errors.

---

# 8. Model Checkpoints

Checkpoints are not stored in GitHub because of their size.

Expected structure:

```text
checkpoints/
└── im2latex_full_plus_school_ft/
    ├── best.pt
    ├── best_ema.pt
    ├── last.pt
    ├── current_bot_best.pt
    └── tokenizer.json
```

Recommended checkpoint for inference:

```text
best_ema.pt
```

---

# 9. Run Training

```bash
python run_train.py
```

Training configuration:

```text
im2latex/config.py
```

Main configuration:

```text
Encoder: ConvNeXt-small
Decoder: Transformer autoregressive decoder
Embedding dimension: 384
Maximum vocabulary size: 11000
Maximum sequence length: 320
Image height: 64
Maximum image width: 512
Optimizer: AdamW
EMA: enabled
Beam search decoding
```

---

# 10. Run Evaluation

```bash
python run_eval.py \
  --ckpt checkpoints/im2latex_full_plus_school_ft/best_ema.pt \
  --tokenizer checkpoints/im2latex_full_plus_school_ft/tokenizer.json \
  --val_csv datasets/im2latex/final_full_plus_school/val.csv \
  --images_dir datasets/im2latex/final_full_plus_school/images \
  --max_samples 1000
```

Current best result:

```text
Char-BLEU = 92.191
ExactMatch = 65.500%
CanonicalExactMatch = 69.500%
EditSimilarity = 94.000%
```

Additional evaluation on school-symbol and simple-formula validation subsets:

```text
ExactMatch = 100%
CanonicalExactMatch = 100%
```

---

# 11. Run Single Image Inference

```bash
python run_infer.py \
  --ckpt checkpoints/im2latex_full_plus_school_ft/best_ema.pt \
  --tokenizer checkpoints/im2latex_full_plus_school_ft/tokenizer.json \
  --image test_images/formula1.png
```

Example output:

```latex
\int_{0}^{1}x^{2}dx
```

---

# 12. Run Telegram Bot

Create `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token
```

Run bot:

```bash
python tgBot/bot.py
```

Bot workflow:

1. user sends image;
2. image is downloaded locally;
3. neural network inference is executed;
4. beam-search decoding generates LaTeX;
5. recognized formula is returned to user.

---

# 13. Error Analysis

Run automated error analysis:

```bash
python analyze_errors.py
```

Typical recognition errors:

- visually similar symbols;
- repeated tokens in long formulas;
- missing braces;
- incorrect superscripts or subscripts;
- formatting differences between equivalent LaTeX expressions.

---

# 14. Current Limitations

Current limitations include:

- blurry photographs;
- Telegram image compression;
- difficult handwritten formulas;
- long nested expressions;
- visually similar symbols;
- domain gap between rendered and real-world images.

Future work focuses on:

- augmentation pipeline improvements;
- render-based validation;
- handwritten robustness;
- decoding optimization;
- real-world dataset collection.

---

# 15. Git Ignore Policy

Large files are not committed to GitHub.

Ignored folders:

```text
.venv/
datasets/
checkpoints/
logs/
__pycache__/
.ipynb_checkpoints/
```

Datasets and checkpoints should be downloaded or generated separately.

---

# 16. Authors

- Bessonov Grigoriy — Telegram bot, GitHub repository, external dataset search
- Levankov Matvey — model training, experiments, checkpoint comparison, dataset experiments
- Ten Aleksei — model training, dataset preparation, synthetic data pipeline, evaluation, documentation

---

# 17. License and Usage

This project was developed as a software team course project.

The system is intended for educational and research purposes.
