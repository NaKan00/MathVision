from pathlib import Path
import os
import pandas as pd


ROOT = Path(".")

FULL_DIR = ROOT / "datasets" / "im2latex" / "final_all_mix"
FULL_IMAGES = FULL_DIR / "images"

OUT_DIR = ROOT / "datasets" / "im2latex" / "final_all_mix_small"
OUT_IMAGES = OUT_DIR / "images"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


TRAIN_COUNTS = {
    "combined": 60000,
    "im2latex": 45000,
    "simple": 15000,
}

VAL_COUNTS = {
    "combined": 3000,
    "im2latex": 2000,
    "simple": 1000,
}


def make_link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return

    os.symlink(src.resolve(), dst, target_is_directory=True)


def sample_by_source(df: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    parts = []

    for source, n in counts.items():
        part = df[df["source"] == source].copy()

        if len(part) == 0:
            raise ValueError(f"No rows for source={source}")

        replace = len(part) < n

        part = part.sample(
            n=n,
            replace=replace,
            random_state=seed,
        )

        parts.append(part)

        print(f"{source:10s}: {len(part)} replace={replace}")

    out = pd.concat(parts, ignore_index=True)

    out = out.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return out


def build_split(split: str, counts: dict, seed: int):
    df = pd.read_csv(FULL_DIR / f"{split}.csv")

    out = sample_by_source(df, counts, seed)

    out_path = OUT_DIR / f"{split}.csv"
    out.to_csv(out_path, index=False)

    print(f"{split} total: {len(out)}")
    print(f"saved: {out_path}")
    print()


def main():
    for source in ["combined", "im2latex", "simple"]:
        make_link(
            FULL_IMAGES / source,
            OUT_IMAGES / source,
        )

    build_split("train", TRAIN_COUNTS, seed=42)
    build_split("val", VAL_COUNTS, seed=43)

    test_path = FULL_DIR / "test.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        test.to_csv(OUT_DIR / "test.csv", index=False)
        print(f"test copied: {len(test)}")

    print("done")


if __name__ == "__main__":
    main()