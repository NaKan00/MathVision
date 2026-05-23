from pathlib import Path
from PIL import Image
import pandas as pd
import shutil

# отключаем защиту PIL только для проверки размеров
Image.MAX_IMAGE_PIXELS = None

DATASET_DIR = Path(r"D:\tgBot\MathVision\datasets\im2latex\combined_math")
IMAGES_DIR = DATASET_DIR / "images"

MAX_PIXELS = 50_000_000      # всё больше 50 млн пикселей удаляем
MAX_SIDE = 12000             # если ширина/высота больше 12000 — удаляем

CSV_FILES = [
    DATASET_DIR / "train.csv",
    DATASET_DIR / "val.csv",
    DATASET_DIR / "test.csv",
]


def is_bad_image(image_name: str):
    path = IMAGES_DIR / str(image_name)

    if not path.exists():
        return True, "missing"

    try:
        with Image.open(path) as img:
            w, h = img.size
            pixels = w * h

            if pixels > MAX_PIXELS:
                return True, f"too_many_pixels: {w}x{h}={pixels}"

            if w > MAX_SIDE or h > MAX_SIDE:
                return True, f"too_large_side: {w}x{h}"

            return False, f"ok: {w}x{h}"

    except Exception as e:
        return True, f"broken: {e}"


def clean_csv(csv_path: Path):
    print(f"\nChecking {csv_path.name}")

    df = pd.read_csv(csv_path)

    good_rows = []
    bad_rows = []

    for i, row in df.iterrows():
        image_name = row["image"]

        bad, reason = is_bad_image(image_name)

        if bad:
            bad_rows.append({
                "image": image_name,
                "formula": row.get("formula", ""),
                "source": row.get("source", ""),
                "reason": reason,
            })

            if len(bad_rows) <= 20:
                print(f"BAD: {image_name} | {reason}")

        else:
            good_rows.append(row)

        if i % 10000 == 0:
            print(f"{csv_path.name}: checked {i}/{len(df)}")

    backup_path = csv_path.with_suffix(".backup.csv")
    shutil.copy2(csv_path, backup_path)

    clean_df = pd.DataFrame(good_rows)
    clean_df.to_csv(csv_path, index=False, encoding="utf-8")

    bad_df = pd.DataFrame(bad_rows)
    bad_report = csv_path.with_name(csv_path.stem + "_bad_images.csv")
    bad_df.to_csv(bad_report, index=False, encoding="utf-8")

    print(f"\n{csv_path.name}")
    print(f"before: {len(df)}")
    print(f"after:  {len(clean_df)}")
    print(f"removed:{len(bad_df)}")
    print(f"backup: {backup_path}")
    print(f"bad report: {bad_report}")


def main():
    for csv_path in CSV_FILES:
        clean_csv(csv_path)

    print("\nDONE. Dataset cleaned.")


if __name__ == "__main__":
    main()