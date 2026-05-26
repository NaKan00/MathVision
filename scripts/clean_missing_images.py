from pathlib import Path
import pandas as pd


ROOT = Path("datasets/im2latex/final_full_plus_school")

for split in ["train", "val"]:
    csv_path = ROOT / f"{split}.csv"

    df = pd.read_csv(csv_path)

    keep_rows = []

    missing = 0

    for _, row in df.iterrows():
        img_path = ROOT / "images" / row["image"]

        if img_path.exists():
            keep_rows.append(row)
        else:
            missing += 1

    cleaned = pd.DataFrame(keep_rows)

    cleaned.to_csv(csv_path, index=False)

    print(split)
    print("before:", len(df))
    print("after :", len(cleaned))
    print("removed missing:", missing)
    print()