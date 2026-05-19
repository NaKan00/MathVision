from pathlib import Path
from collections import Counter
import re

import pandas as pd
from PIL import Image
from tqdm import tqdm


TRAIN_CSV = Path("datasets/im2latex/train.csv")
VAL_CSV = Path("datasets/im2latex/val.csv")
IMAGES_DIR = Path("datasets/im2latex/images/formula_images_processed")
OUT_DIR = Path("dataset_analysis")
OUT_DIR.mkdir(exist_ok=True)


LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")


def get_image_path(image_name: str) -> Path:
    return IMAGES_DIR / image_name


def analyze_split(csv_path: Path, split_name: str):
    print(f"\n=== {split_name.upper()} ===")

    df = pd.read_csv(csv_path)

    if "formula" not in df.columns or "image" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: formula, image")

    df["formula"] = df["formula"].astype(str)
    df["formula_len_chars"] = df["formula"].str.len()
    df["formula_len_tokens_rough"] = df["formula"].str.split().str.len()

    widths = []
    heights = []
    missing_images = []
    broken_images = []

    for image_name in tqdm(df["image"], desc=f"images {split_name}"):
        path = get_image_path(image_name)

        if not path.exists():
            widths.append(None)
            heights.append(None)
            missing_images.append(image_name)
            continue

        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            widths.append(None)
            heights.append(None)
            broken_images.append(image_name)

    df["image_width"] = widths
    df["image_height"] = heights
    df["aspect_ratio"] = df["image_width"] / df["image_height"]

    print("rows:", len(df))
    print("missing images:", len(missing_images))
    print("broken images:", len(broken_images))

    print("\nFormula length chars:")
    print(df["formula_len_chars"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nImage width:")
    print(df["image_width"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nImage height:")
    print(df["image_height"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nAspect ratio:")
    print(df["aspect_ratio"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    latex_commands = Counter()
    for formula in df["formula"]:
        latex_commands.update(LATEX_COMMAND_RE.findall(formula))

    top_commands = pd.DataFrame(
        latex_commands.most_common(50),
        columns=["command", "count"],
    )

    long_threshold_95 = df["formula_len_chars"].quantile(0.95)
    long_threshold_99 = df["formula_len_chars"].quantile(0.99)

    long_95 = df[df["formula_len_chars"] > long_threshold_95].copy()
    long_99 = df[df["formula_len_chars"] > long_threshold_99].copy()

    wide_95 = df[df["image_width"] > df["image_width"].quantile(0.95)].copy()
    extreme_aspect = df[df["aspect_ratio"] > df["aspect_ratio"].quantile(0.99)].copy()

    df.to_csv(OUT_DIR / f"{split_name}_analysis_full.csv", index=False)
    top_commands.to_csv(OUT_DIR / f"{split_name}_top_latex_commands.csv", index=False)
    long_95.to_csv(OUT_DIR / f"{split_name}_long_formulas_top5_percent.csv", index=False)
    long_99.to_csv(OUT_DIR / f"{split_name}_long_formulas_top1_percent.csv", index=False)
    wide_95.to_csv(OUT_DIR / f"{split_name}_wide_images_top5_percent.csv", index=False)
    extreme_aspect.to_csv(OUT_DIR / f"{split_name}_extreme_aspect_top1_percent.csv", index=False)

    print("\nTop LaTeX commands:")
    print(top_commands.head(20))

    print("\nSaved files:")
    print(OUT_DIR / f"{split_name}_analysis_full.csv")
    print(OUT_DIR / f"{split_name}_top_latex_commands.csv")
    print(OUT_DIR / f"{split_name}_long_formulas_top5_percent.csv")
    print(OUT_DIR / f"{split_name}_long_formulas_top1_percent.csv")
    print(OUT_DIR / f"{split_name}_wide_images_top5_percent.csv")
    print(OUT_DIR / f"{split_name}_extreme_aspect_top1_percent.csv")

    return df


def compare_train_val(train_df, val_df):
    print("\n=== TRAIN / VAL COMPARISON ===")

    summary = pd.DataFrame({
        "train": [
            len(train_df),
            train_df["formula_len_chars"].median(),
            train_df["formula_len_chars"].quantile(0.95),
            train_df["formula_len_chars"].max(),
            train_df["image_width"].median(),
            train_df["image_width"].quantile(0.95),
            train_df["image_width"].max(),
        ],
        "val": [
            len(val_df),
            val_df["formula_len_chars"].median(),
            val_df["formula_len_chars"].quantile(0.95),
            val_df["formula_len_chars"].max(),
            val_df["image_width"].median(),
            val_df["image_width"].quantile(0.95),
            val_df["image_width"].max(),
        ],
    }, index=[
        "rows",
        "formula_len_median",
        "formula_len_p95",
        "formula_len_max",
        "image_width_median",
        "image_width_p95",
        "image_width_max",
    ])

    print(summary)
    summary.to_csv(OUT_DIR / "train_val_summary.csv")


def main():
    train_df = analyze_split(TRAIN_CSV, "train")
    val_df = analyze_split(VAL_CSV, "val")
    compare_train_val(train_df, val_df)

    print("\n=== RECOMMENDATION CHECKS ===")

    train_len_95 = train_df["formula_len_chars"].quantile(0.95)
    train_width_95 = train_df["image_width"].quantile(0.95)
    train_ar_99 = train_df["aspect_ratio"].quantile(0.99)

    print(f"Suggested formula filtering threshold, p95: len <= {int(train_len_95)}")
    print(f"Suggested wide image attention threshold, p95 width: {int(train_width_95)}")
    print(f"Extreme aspect ratio threshold, p99: aspect_ratio > {train_ar_99:.2f}")

    


if __name__ == "__main__":
    main()