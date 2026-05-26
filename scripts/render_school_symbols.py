import csv
import tempfile
import subprocess
from pathlib import Path

from PIL import Image


CSV_PATH = Path("datasets/school_symbols/formulas.csv")

OUT_DIR = Path("datasets/school_symbols/images")
OUT_CSV = Path("datasets/school_symbols/train.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)


LATEX_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
\[
%s
\]
\end{document}
"""


def render_formula(formula: str, out_png: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tex_path = tmpdir / "eq.tex"
        dvi_path = tmpdir / "eq.dvi"
        png_path = tmpdir / "eq.png"

        tex_path.write_text(LATEX_TEMPLATE % formula, encoding="utf-8")

        subprocess.run(
            [
                "latex",
                "-interaction=nonstopmode",
                "eq.tex",
            ],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            [
                "dvipng",
                "-T",
                "tight",
                "-D",
                "160",
                "-bg",
                "Transparent",
                "-o",
                str(png_path),
                str(dvi_path),
            ],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        if not png_path.exists():
            raise FileNotFoundError(png_path)

        img = Image.open(png_path).convert("L")
        img.save(out_png)


def main():
    rows = []
    failed = 0

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            formula = row["formula"]

            img_name = f"school_{idx:06d}.png"
            out_png = OUT_DIR / img_name

            try:
                render_formula(formula, out_png)

                rows.append({
                    "image": img_name,
                    "formula": formula,
                })

            except Exception as e:
                failed += 1
                print("FAILED:", idx, formula)
                print(repr(e))
                continue

            if idx % 500 == 0:
                print(idx)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "formula"],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("saved:", OUT_CSV)
    print("num samples:", len(rows))
    print("failed samples:", failed)


if __name__ == "__main__":
    main()