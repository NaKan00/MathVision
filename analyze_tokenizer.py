from pathlib import Path
from collections import Counter

import pandas as pd

from im2latex.config import TRAIN_CSV, VOCAB_MIN_FREQ, VOCAB_MAX_SIZE
from im2latex.data.tokenizer import Tokenizer, latex_tokenize


CHECK_TOKENS = [
    r"\frac",
    r"\partial",
    r"\operatorname",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\sqrt",
    r"\int",
    r"\sum",

    r"\left(",
    r"\right)",
    r"\left[",
    r"\right]",
    r"\left\{",
    r"\right\}",
]


def main():
    df = pd.read_csv(TRAIN_CSV)
    formulas = df["formula"].astype(str).tolist()

    tok = Tokenizer.build(
        formulas,
        min_freq=VOCAB_MIN_FREQ,
        max_size=VOCAB_MAX_SIZE,
    )

    counter = Counter()
    lengths = []

    for f in formulas:
        tokens = latex_tokenize(f)
        counter.update(tokens)
        lengths.append(len(tokens))

    print("=== Tokenizer audit ===")
    print("vocab size:", len(tok.vocab.itos))
    print("num formulas:", len(formulas))
    print("median token len:", sorted(lengths)[len(lengths) // 2])
    print("max token len:", max(lengths))

    print("\n=== Important tokens ===")
    for t in CHECK_TOKENS:
        print(
            f"{t:15s}",
            "IN_VOCAB" if t in tok.vocab.stoi else "MISSING",
            "freq=",
            counter.get(t, 0),
        )

    print("\n=== Top 50 tokens ===")
    for token, count in counter.most_common(50):
        print(f"{token:20s} {count}")


if __name__ == "__main__":
    main()