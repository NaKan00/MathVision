from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image, ImageDraw


# =========================================
# PATHS
# =========================================

ROOT = Path(r"D:\tgBot\MathVision\datasets\im2latex")

HME100K_DIR = ROOT / "HME100K"
MATHWRITING_DIR = ROOT / "MathWriting"
CROHME_DIR = ROOT / "CROHME"

OUT_DIR = ROOT / "combined_math"

SEED = 42
VAL_RATIO = 0.08
TEST_RATIO = 0.08


# =========================================
# HELPERS
# =========================================

random.seed(SEED)

OUT_IMAGES = OUT_DIR / "images"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


def canonicalize(s: str):
    return " ".join(str(s).strip().split())


def save_copy(src: Path, dst: Path):
    img = Image.open(src).convert("L")
    img.save(dst)


def safe_name(prefix, idx):
    return f"{prefix}_{idx:08d}.png"


# =========================================
# HME100K
# =========================================

def load_hme100k():
    rows = []
    idx = 0

    label_files = list(HME100K_DIR.rglob("*labels.txt"))

    for lf in label_files:

        text = lf.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        image_dir = lf.parent / lf.stem.replace("_labels", "_images")

        for line in text:

            parts = line.strip().split(maxsplit=1)

            if len(parts) != 2:
                continue

            img_name, formula = parts

            src_img = image_dir / img_name

            if not src_img.exists():
                continue

            out_name = safe_name("hme100k", idx)

            save_copy(
                src_img,
                OUT_IMAGES / out_name,
            )

            rows.append({
                "image": out_name,
                "formula": canonicalize(formula),
                "source": "hme100k",
            })

            idx += 1

    print("HME100K:", len(rows))
    return rows


# =========================================
# MATHWRITING
# =========================================

def load_mathwriting():
    rows = []
    idx = 0

    png_files = list(MATHWRITING_DIR.rglob("*.png"))

    for png in png_files:

        txt = png.with_suffix(".txt")

        if not txt.exists():
            continue

        formula = txt.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if not formula:
            continue

        out_name = safe_name("mathwriting", idx)

        save_copy(
            png,
            OUT_IMAGES / out_name,
        )

        rows.append({
            "image": out_name,
            "formula": canonicalize(formula),
            "source": "mathwriting",
        })

        idx += 1

    print("MathWriting:", len(rows))
    return rows


# =========================================
# CROHME
# =========================================

def strip_ns(tag):
    return tag.split("}")[-1]


def parse_trace(trace_text):

    pts = []

    chunks = trace_text.strip().split(",")

    for ch in chunks:

        nums = ch.strip().split()

        if len(nums) >= 2:

            try:
                x = float(nums[0])
                y = float(nums[1])

                pts.append((x, y))

            except:
                pass

    return pts


def extract_formula(root):

    for elem in root.iter():

        tag = strip_ns(elem.tag).lower()

        if tag == "annotation":

            ann_type = elem.attrib.get(
                "type",
                "",
            ).lower()

            if ann_type in [
                "truth",
                "latex",
                "normalizedtruth",
            ]:

                text = elem.text

                if text:
                    return canonicalize(text)

    return ""


def render_inkml(inkml_path, out_path):

    tree = ET.parse(inkml_path)
    root = tree.getroot()

    traces = []

    for elem in root.iter():

        if strip_ns(elem.tag).lower() == "trace":

            pts = parse_trace(elem.text)

            if pts:
                traces.append(pts)

    if not traces:
        return ""

    formula = extract_formula(root)

    all_pts = [p for t in traces for p in t]

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    scale = 2.0
    pad = 20

    w = int((max_x - min_x) * scale + 2 * pad)
    h = int((max_y - min_y) * scale + 2 * pad)

    img = Image.new("L", (w, h), 255)

    draw = ImageDraw.Draw(img)

    for tr in traces:

        pts = []

        for x, y in tr:

            xx = (x - min_x) * scale + pad
            yy = (y - min_y) * scale + pad

            pts.append((xx, yy))

        if len(pts) >= 2:
            draw.line(
                pts,
                fill=0,
                width=3,
            )

    img.save(out_path)

    return formula


def load_crohme():

    rows = []
    idx = 0

    inkml_files = list(CROHME_DIR.rglob("*.inkml"))

    for inkml in inkml_files:

        out_name = safe_name("crohme", idx)

        out_img = OUT_IMAGES / out_name

        try:

            formula = render_inkml(
                inkml,
                out_img,
            )

            if not formula:
                continue

            rows.append({
                "image": out_name,
                "formula": formula,
                "source": "crohme",
            })

            idx += 1

        except Exception as e:

            print("CROHME ERROR:", inkml)
            print(e)

    print("CROHME:", len(rows))
    return rows


# =========================================
# SPLIT
# =========================================

def split_rows(rows):

    by_source = {}

    for r in rows:

        by_source.setdefault(
            r["source"],
            [],
        ).append(r)

    train = []
    val = []
    test = []

    for source, src_rows in by_source.items():

        random.shuffle(src_rows)

        n = len(src_rows)

        n_test = int(n * TEST_RATIO)
        n_val = int(n * VAL_RATIO)

        test.extend(src_rows[:n_test])

        val.extend(
            src_rows[n_test:n_test + n_val]
        )

        train.extend(
            src_rows[n_test + n_val:]
        )

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# =========================================
# MAIN
# =========================================

def main():

    all_rows = []

    all_rows.extend(load_hme100k())
    all_rows.extend(load_mathwriting())
    all_rows.extend(load_crohme())

    print("TOTAL:", len(all_rows))

    train, val, test = split_rows(all_rows)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(train).to_csv(
        OUT_DIR / "train.csv",
        index=False,
    )

    pd.DataFrame(val).to_csv(
        OUT_DIR / "val.csv",
        index=False,
    )

    pd.DataFrame(test).to_csv(
        OUT_DIR / "test.csv",
        index=False,
    )

    print()
    print("DONE")
    print("train:", len(train))
    print("val:", len(val))
    print("test:", len(test))


if __name__ == "__main__":
    main()