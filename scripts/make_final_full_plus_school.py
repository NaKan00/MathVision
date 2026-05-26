from pathlib import Path
import os
import pandas as pd


ROOT = Path(".")

FULL_DIR = ROOT / "datasets" / "im2latex" / "final_all_mix"
SCHOOL_DIR = ROOT / "datasets" / "school_symbols"

OUT_DIR = ROOT / "datasets" / "im2latex" / "final_full_plus_school"
OUT_IMAGES = OUT_DIR / "images"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


def make_link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return

    os.symlink(src.resolve(), dst, target_is_directory=True)


def add_source_prefix(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "image" not in df.columns:
        raise ValueError("No image column")

    if "formula" not in df.columns:
        raise ValueError("No formula column")

    df = df[["image", "formula"]].copy()

    df["image"] = df["image"].astype(str)
    df["formula"] = df["formula"].astype(str)

    df = df.dropna()
    df = df[df["image"].str.len() > 0]
    df = df[df["formula"].str.len() > 0]

    df["image"] = source + "/" + df["image"]
    df["source"] = source

    return df


def build_split(split: str):
    full_path = FULL_DIR / f"{split}.csv"

    if not full_path.exists():
        raise FileNotFoundError(full_path)

    full = pd.read_csv(full_path)

    rows = [full]

    if split == "train":
        school_path = SCHOOL_DIR / "train_school.csv"
    elif split == "val":
        school_path = SCHOOL_DIR / "val_school.csv"
    else:
        school_path = None

    if school_path is not None:
        if not school_path.exists():
            raise FileNotFoundError(school_path)

        school = pd.read_csv(school_path)
        school = add_source_prefix(school, "school")
        rows.append(school)

    mixed = pd.concat(rows, ignore_index=True)

    mixed = mixed.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    out_path = OUT_DIR / f"{split}.csv"
    mixed.to_csv(out_path, index=False)

    print(f"{split}: {len(mixed)}")

    if "source" in mixed.columns:
        print(mixed["source"].value_counts())

    print("saved:", out_path)
    print()


def main():
    for source in ["combined", "im2latex", "simple"]:
        make_link(
            FULL_DIR / "images" / source,
            OUT_IMAGES / source,
        )

    make_link(
        SCHOOL_DIR / "images",
        OUT_IMAGES / "school",
    )

    build_split("train")
    build_split("val")

    test_path = FULL_DIR / "test.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        out_test = OUT_DIR / "test.csv"
        test.to_csv(out_test, index=False)

        print("test copied:", len(test))
        print("saved:", out_test)

    print("final images dir:", OUT_IMAGES)
    print("done")


if __name__ == "__main__":
    main()