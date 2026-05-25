from pathlib import Path
import pandas as pd
import os


ROOT = Path(".")

OUT_DIR = ROOT / "datasets" / "im2latex" / "final_all_mix"
OUT_IMAGES = OUT_DIR / "images"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {
        "name": "im2latex",
        "train_csv": ROOT / "datasets" / "im2latex" / "train.csv",
        "val_csv": ROOT / "datasets" / "im2latex" / "val.csv",
        "images_dir": ROOT / "datasets" / "im2latex" / "images",
    },
    {
        "name": "simple",
        "train_csv": ROOT / "datasets" / "simple_formulas" / "train_simple.csv",
        "val_csv": ROOT / "datasets" / "simple_formulas" / "val_simple.csv",
        "images_dir": ROOT / "datasets" / "simple_formulas" / "images",
    },
    {
        "name": "combined",
        "train_csv": ROOT / "datasets" / "im2latex" / "combined" / "train.csv",
        "val_csv": ROOT / "datasets" / "im2latex" / "combined" / "val.csv",
        "images_dir": ROOT / "datasets" / "im2latex" / "combined" / "images",
    },
]


def make_link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return

    src_abs = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    os.symlink(src_abs, dst, target_is_directory=True)


def normalize_df(csv_path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "image" not in df.columns:
        raise ValueError(f"No image column in {csv_path}")

    if "formula" not in df.columns:
        raise ValueError(f"No formula column in {csv_path}")

    df = df[["image", "formula"]].copy()

    df["image"] = df["image"].astype(str)
    df["formula"] = df["formula"].astype(str)

    df = df.dropna()

    df = df[df["image"].str.len() > 0]
    df = df[df["formula"].str.len() > 0]

    df["image"] = source_name + "/" + df["image"]
    df["source"] = source_name

    return df


def build_split(split: str):
    rows = []

    for ds in DATASETS:
        csv_path = ds[f"{split}_csv"]
        source_name = ds["name"]

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = normalize_df(csv_path, source_name)

        rows.append(df)

        make_link(
            src=ds["images_dir"],
            dst=OUT_IMAGES / source_name,
        )

        print(f"{split:5s} {source_name:10s}: {len(df)}")

    mixed = pd.concat(rows, ignore_index=True)

    mixed = mixed.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    out_csv = OUT_DIR / f"{split}.csv"

    mixed.to_csv(out_csv, index=False)

    print(f"{split:5s} total: {len(mixed)}")
    print(f"saved: {out_csv}")
    print()


def main():
    build_split("train")
    build_split("val")

    print("final images dir:", OUT_IMAGES)
    print("done")


if __name__ == "__main__":
    main()