import pandas as pd
from pathlib import Path


SRC = Path("datasets/school_symbols/train.csv")

OUT_TRAIN = Path("datasets/school_symbols/train_school.csv")
OUT_VAL = Path("datasets/school_symbols/val_school.csv")


def main():
    df = pd.read_csv(SRC)

    df = df[["image", "formula"]].copy()
    df["image"] = df["image"].astype(str)
    df["formula"] = df["formula"].astype(str)

    df = df.dropna()
    df = df[df["image"].str.len() > 0]
    df = df[df["formula"].str.len() > 0]

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_val = max(2000, int(len(df) * 0.05))

    val = df.iloc[:n_val]
    train = df.iloc[n_val:]

    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)

    print("train:", len(train))
    print("val:", len(val))
    print("saved:", OUT_TRAIN)
    print("saved:", OUT_VAL)


if __name__ == "__main__":
    main()