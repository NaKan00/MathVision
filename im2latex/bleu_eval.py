import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from im2latex.utils import load_tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex


def normalize_tex(s: str) -> str:
    return " ".join(str(s).strip().split())


def load_checkpoint(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    if "model_state" in obj and isinstance(obj["model_state"], dict):
        return obj
    if "model" in obj and isinstance(obj["model"], dict):
        obj["model_state"] = obj.pop("model")
        return obj
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj["model_state"] = obj.pop("state_dict")
        return obj
    if all(isinstance(k, str) for k in obj.keys()):
        return {"model_state": obj}
    raise ValueError("Cannot extract model_state from checkpoint")


@torch.no_grad()
def beam_decode_single(
    model: Img2Latex,
    x_single: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int = 5,
    max_len: int = 160,
    length_penalty_alpha: float = 0.6,
    no_repeat_ngram: int = 3,
):
    """
    Beam search на 1 картинку. Возвращает тензор токенов [T].
    """
    device = x_single.device
    model.eval()

    beams = [(torch.tensor([bos_id], device=device, dtype=torch.long), 0.0, False)]

    def lp(length: int) -> float:
        return ((5.0 + length) / 6.0) ** length_penalty_alpha

    def block_repeated_ngrams(log_probs_row: torch.Tensor, seq: list[int], n: int):
        if n <= 0 or len(seq) < n:
            return log_probs_row
        prefix = seq[-(n - 1) :]
        banned = set()
        for i in range(len(seq) - n + 1):
            if seq[i : i + (n - 1)] == prefix:
                banned.add(seq[i + (n - 1)])
        if banned:
            idx = torch.tensor(list(banned), device=log_probs_row.device, dtype=torch.long)
            log_probs_row.index_fill_(0, idx, -1e9)
        return log_probs_row

    for _ in range(max_len - 1):
        if all(b[2] for b in beams):
            break

        all_candidates = []
        for seq, score, finished in beams:
            if finished:
                all_candidates.append((seq, score, True))
                continue

            tgt = seq.unsqueeze(0)
            logits = model(x_single, tgt)
            next_logits = logits[:, -1, :].squeeze(0)

            next_logits[pad_id] = -1e9

            log_probs = torch.log_softmax(next_logits, dim=-1)
            log_probs = block_repeated_ngrams(log_probs, seq.tolist(), no_repeat_ngram)

            topk = torch.topk(log_probs, k=max(1, beam_size))
            for token_id, token_lp in zip(topk.indices.tolist(), topk.values.tolist()):
                token_id = int(token_id)
                new_seq = torch.cat(
                    [seq, torch.tensor([token_id], device=device, dtype=torch.long)],
                    dim=0,
                )
                new_score = score + float(token_lp)
                new_finished = (token_id == eos_id)
                all_candidates.append((new_seq, new_score, new_finished))

        all_candidates.sort(key=lambda t: (t[1] / lp(len(t[0]))), reverse=True)
        beams = all_candidates[: max(1, beam_size)]

    best = max(beams, key=lambda t: (t[1] / lp(len(t[0]))))
    return best[0]

def char_ngrams(s: str, n: int):
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def corpus_char_bleu(preds, refs, max_n=4, smooth=1.0):
    """
    BLEU по символам (0..100).
    """
    import math
    from collections import Counter

    p_ns = []
    pred_len = 0
    ref_len = 0

    for n in range(1, max_n + 1):
        match = 0
        total = 0

        for pred, ref in zip(preds, refs):
            if n == 1:
                pred_len += len(pred)
                ref_len += len(ref)

            p_ngr = Counter(char_ngrams(pred, n))
            r_ngr = Counter(char_ngrams(ref, n))

            total += sum(p_ngr.values())
            for ng, c in p_ngr.items():
                match += min(c, r_ngr.get(ng, 0))

        p_n = (match + smooth) / (total + smooth) if total > 0 else 0.0
        p_ns.append(p_n)

    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))

    score = bp * math.exp(sum(math.log(p) for p in p_ns) / max_n)
    return 100.0 * score


def exact_match(preds, refs):
    eq = sum(1 for p, r in zip(preds, refs) if p == r)
    return 100.0 * eq / max(1, len(refs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--val_csv", default="datasets/im2latex/val.csv")
    ap.add_argument("--images_dir", default="datasets/im2latex/images/formula_images_processed")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_samples", type=int, default=1000, help="0 = all")

    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--lp", type=float, default=0.6, help="length penalty alpha")
    ap.add_argument("--no_repeat_ngram", type=int, default=3)

    ap.add_argument("--print_examples", type=int, default=5)

    ap.add_argument("--tokenizer", default=None)

    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(str(ckpt_path))

    tok_path = Path(args.tokenizer) if args.tokenizer is not None else (ckpt_path.parent / "tokenizer.json")
    tok = load_tokenizer(tok_path)

    d_model = int(ckpt.get("d_model", 256))
    encoder_variant = str(ckpt.get("encoder_variant", "small"))

    val_df = pd.read_csv(args.val_csv)
    tmp_csv = None
    if args.max_samples and args.max_samples > 0:
        val_df = val_df.iloc[: args.max_samples].copy()
        tmp_csv = Path(".bleu_tmp_val.csv")
        val_df.to_csv(tmp_csv, index=False)
        val_csv_path = str(tmp_csv)
    else:
        val_csv_path = args.val_csv

    ds = Im2LatexDataset(
        val_csv_path,
        args.images_dir,
        tok,
        build_image_transform(height=64, max_width=384),
        max_len=args.max_len,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
    )

    model = Img2Latex(
        vocab_size=len(tok.vocab.itos),
        pad_id=tok.vocab.pad,
        d_model=d_model,
        encoder_variant=encoder_variant,
        encoder_pretrained=False,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded ckpt: d_model={d_model}, encoder_variant={encoder_variant}")
    print(f"Tokenizer: {tok_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print("Missing key names:")
        for k in missing:
            print(f"  - {k}")

    preds, refs = [], []
    printed = 0

    for x, y in tqdm(dl, desc="Eval"):
        x = x.to(device)
        y = y.to(device)

        for i in range(x.size(0)):
            seq = beam_decode_single(
                model,
                x[i : i + 1],
                bos_id=tok.vocab.bos,
                eos_id=tok.vocab.eos,
                pad_id=tok.vocab.pad,
                beam_size=max(1, args.beam),
                max_len=args.max_len,
                length_penalty_alpha=args.lp,
                no_repeat_ngram=args.no_repeat_ngram,
            )

            prd = normalize_tex(tok.decode(seq.tolist(), skip_special=True))
            ref = normalize_tex(tok.decode(y[i].tolist(), skip_special=True))

            preds.append(prd)
            refs.append(ref)

            if printed < args.print_examples:
                print("REF:", ref)
                print("PRD:", prd)
                print("-" * 60)
                printed += 1

    bleu_char = corpus_char_bleu(preds, refs, max_n=4, smooth=1.0)
    em = exact_match(preds, refs)

    print(f"Char-BLEU = {bleu_char:.3f}")
    print(f"ExactMatch = {em:.3f}%")

    if tmp_csv is not None and tmp_csv.exists():
        tmp_csv.unlink()


if __name__ == "__main__":
    main()