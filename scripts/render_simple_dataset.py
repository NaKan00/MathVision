import csv
import tempfile
import subprocess
from pathlib import Path

from PIL import Image


CSV_PATH = Path("datasets/simple_formulas/simple_formulas.csv")

OUT_DIR = Path("datasets/simple_formulas/images")
OUT_CSV = Path("datasets/simple_formulas/train.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)


LATEX_TEMPLATE = r"""
\documentclass[preview,border=3pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
$
%s
$
\end{document}
"""


def render_formula(formula: str, out_png: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tex_path = tmpdir / "eq.tex"
        tex_path.write_text(LATEX_TEMPLATE % formula, encoding="utf-8")

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "eq.tex"],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "eq.pdf", "eq"],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        png_path = tmpdir / "eq.png"

        if not png_path.exists():
            raise FileNotFoundError(f"PNG was not created: {png_path}")

        img = Image.open(png_path).convert("L")
        img.save(out_png)


def main():
    rows = []
    failed = 0

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            formula = row["formula"]

            img_name = f"simple_{idx:06d}.png"
            out_png = OUT_DIR / img_name

            try:
                render_formula(formula, out_png)

                rows.append({
                    "image": img_name,
                    "formula": formula,
                })

            except Exception as e:
                failed += 1

                if failed <= 20:
                    print("FAILED:", idx, formula)
                    print(repr(e))

                continue

            if idx % 100 == 0:
                print(idx)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "formula"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved dataset: {OUT_CSV}")
    print(f"num samples: {len(rows)}")
    print(f"failed samples: {failed}")


if __name__ == "__main__":
    main()