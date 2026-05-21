import csv
import argparse
from pathlib import Path

from im2latex.bleu_eval import (
    load_checkpoint,
    beam_decode_single,
    normalize_tex,
    edit_distance,
)
from im2latex.canonical import canonicalize_latex

import torch
from tqdm import tqdm

from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex
from im2latex.utils import load_tokenizer


def edit_similarity(pred: str, ref: str) -> float:
    denom = max(len(pred), len(ref), 1)
    return 100.0 * (1.0 - edit_distance(pred, ref) / denom)


def classify_error(pred: str, ref: str) -> str:
    if pred == ref:
        return "exact"

    if canonicalize_latex(pred) == canonicalize_latex(ref):
        return "canonical_equivalent"

    sim = edit_similarity(pred, ref)

    if sim >= 95:
        return "almost_correct"
    if len(pred) < len(ref) * 0.75:
        return "too_short"
    if len(pred) > len(ref) * 1.25:
        return "too_long"
    if pred.count("{") != pred.count("}") or ref.count("{") != ref.count("}"):
        return "brace_issue"
    if "!" in ref and "!" not in pred:
        return "missing_factorial"
    if "^" in ref and "^" not in pred:
        return "missing_power"
    if "_" in ref and "_" not in pred:
        return "missing_subscript"

    return "semantic_or_token_error"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default="checkpoints/im2latex_convnext/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/im2latex_convnext/tokenizer.json")
    ap.add_argument("--val_csv", default="datasets/im2latex/val.csv")
    ap.add_argument("--images_dir", default="datasets/im2latex/images/formula_images_processed")
    ap.add_argument("--out", default="error_analysis.csv")

    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--repeat_penalty", type=float, default=1.0)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--min_len", type=int, default=4)

    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max_width", type=int, default=512)

    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = load_checkpoint(args.ckpt)
    tok = load_tokenizer(args.tokenizer)

    ds = Im2LatexDataset(
        args.val_csv,
        args.images_dir,
        tok,
        build_image_transform(
            height=args.height,
            max_width=args.max_width,
        ),
        max_len=args.max_len,
    )

    if args.max_samples and args.max_samples > 0:
        ds.rows = ds.rows[: args.max_samples]

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
    )

    model = Img2Latex(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
        d_model=int(ckpt.get("d_model", 256)),
        encoder_variant=str(ckpt.get("encoder_variant", "small")),
        encoder_pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    rows = []
    exact = 0
    canonical_exact = 0

    idx_global = 0

    for x, y, image_pad_mask in tqdm(dl, desc="Analyze"):
        x = x.to(device)
        y = y.to(device)
        image_pad_mask = image_pad_mask.to(device)

        for i in range(x.size(0)):
            seq = beam_decode_single(
                model,
                x[i: i + 1],
                image_pad_mask[i: i + 1],
                bos_id=tok.vocab.bos,
                eos_id=tok.vocab.eos,
                beam_size=args.beam,
                max_len=args.max_len,
                repeat_penalty=args.repeat_penalty,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                min_len=args.min_len,
            )

            pred = normalize_tex(tok.decode(seq.tolist(), skip_special=True))
            ref = normalize_tex(tok.decode(y[i].tolist(), skip_special=True))

            is_exact = pred == ref
            is_canonical = canonicalize_latex(pred) == canonicalize_latex(ref)
            sim = edit_similarity(pred, ref)
            error_type = classify_error(pred, ref)

            if is_exact:
                exact += 1
            if is_canonical:
                canonical_exact += 1

            image_name = ds.rows[idx_global].get("image", "")

            rows.append(
                {
                    "idx": idx_global,
                    "image": image_name,
                    "exact": int(is_exact),
                    "canonical_exact": int(is_canonical),
                    "edit_similarity": round(sim, 3),
                    "error_type": error_type,
                    "ref_len": len(ref),
                    "pred_len": len(pred),
                    "ref": ref,
                    "pred": pred,
                    "ref_canonical": canonicalize_latex(ref),
                    "pred_canonical": canonicalize_latex(pred),
                }
            )

            idx_global += 1

    out_path = Path(args.out)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image",
                "exact",
                "canonical_exact",
                "edit_similarity",
                "error_type",
                "ref_len",
                "pred_len",
                "ref",
                "pred",
                "ref_canonical",
                "pred_canonical",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n = max(len(rows), 1)

    print(f"Saved: {out_path}")
    print(f"Samples: {len(rows)}")
    print(f"ExactMatch: {100.0 * exact / n:.3f}%")
    print(f"CanonicalExactMatch: {100.0 * canonical_exact / n:.3f}%")

    counts = {}
    for r in rows:
        counts[r["error_type"]] = counts.get(r["error_type"], 0) + 1

    print("\nError types:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k:25s} {v:5d}  {100.0 * v / n:6.2f}%")


if __name__ == "__main__":
    main()