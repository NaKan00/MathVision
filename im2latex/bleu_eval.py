import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex
from im2latex.utils import load_tokenizer


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
    image_pad_mask_single: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 5,
    max_len: int = 160,
    length_penalty_alpha: float = 0.6,
    repeat_penalty: float = 1.0,
):
    device = x_single.device
    model.eval()

    memory, memory_key_padding_mask = model.encoder(
        x_single,
        image_pad_mask=image_pad_mask_single,
    )

    beams = [
        (
            torch.tensor([[bos_id]], device=device, dtype=torch.long),
            0.0,
            False,
        )
    ]

    def lp(length: int) -> float:
        return ((5.0 + length) / 6.0) ** length_penalty_alpha

    for _ in range(max_len - 1):
        if all(b[2] for b in beams):
            break

        all_candidates = []

        for seq, score, finished in beams:
            if finished:
                all_candidates.append((seq, score, True))
                continue

            logits = model.decoder(
                seq,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            next_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_logits, dim=-1).squeeze(0)

            if repeat_penalty and repeat_penalty > 1.0:
                for prev_id in seq[0].tolist():
                    if prev_id not in {bos_id, eos_id}:
                        log_probs[prev_id] /= repeat_penalty

            topk = torch.topk(log_probs, k=beam_size)

            for token_id, token_lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_seq = torch.cat(
                    [
                        seq,
                        torch.tensor(
                            [[token_id]],
                            device=device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )

                new_score = score + float(token_lp)
                new_finished = token_id == eos_id

                all_candidates.append((new_seq, new_score, new_finished))

        all_candidates.sort(
            key=lambda t: t[1] / lp(t[0].shape[1]),
            reverse=True,
        )

        beams = all_candidates[:beam_size]

    best = max(
        beams,
        key=lambda t: t[1] / lp(t[0].shape[1]),
    )

    return best[0][0]


def char_ngrams(s: str, n: int):
    if len(s) < n:
        return []
    return [s[i: i + n] for i in range(len(s) - n + 1)]


def corpus_char_bleu(preds, refs, max_n=4, smooth=1.0):
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

    bp = 1.0 if pred_len > ref_len else math.exp(
        1.0 - (ref_len / max(pred_len, 1))
    )

    score = bp * math.exp(sum(math.log(p) for p in p_ns) / max_n)
    return 100.0 * score


def exact_match(preds, refs):
    eq = sum(1 for p, r in zip(preds, refs) if p == r)
    return 100.0 * eq / max(1, len(refs))


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i

        for j in range(1, m + 1):
            temp = dp[j]

            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])

            prev = temp

    return dp[m]


def normalized_edit_similarity(preds, refs):
    scores = []

    for p, r in zip(preds, refs):
        denom = max(len(p), len(r), 1)
        dist = edit_distance(p, r)
        scores.append(1.0 - dist / denom)

    return 100.0 * sum(scores) / max(1, len(scores))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default="checkpoints/im2latex_convnext/last.pt")
    ap.add_argument("--tokenizer", default="checkpoints/im2latex_convnext/tokenizer.json")
    ap.add_argument("--val_csv", default="datasets/im2latex/val.csv")
    ap.add_argument("--images_dir", default="datasets/im2latex/images/formula_images_processed")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--repeat_penalty", type=float, default=1.0)
    ap.add_argument("--print_examples", type=int, default=5)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max_width", type=int, default=384)

    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt_path = Path(args.ckpt)
    tokenizer_path = Path(args.tokenizer)

    ckpt = load_checkpoint(str(ckpt_path))
    tok = load_tokenizer(tokenizer_path)

    d_model = int(ckpt.get("d_model", 256))
    encoder_variant = str(ckpt.get("encoder_variant", "small"))

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
        d_model=d_model,
        encoder_variant=encoder_variant,
        encoder_pretrained=False,
    ).to(device)

    missing, unexpected = model.load_state_dict(
        ckpt["model_state"],
        strict=False,
    )

    print(f"Loaded ckpt: {ckpt_path}")
    print(f"Loaded tokenizer: {tokenizer_path}")
    print(f"d_model={d_model}, encoder_variant={encoder_variant}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    preds, refs = [], []
    printed = 0

    for x, y, image_pad_mask in tqdm(dl, desc="Eval"):
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
                beam_size=max(1, args.beam),
                max_len=args.max_len,
                repeat_penalty=args.repeat_penalty,
            )

            prd = normalize_tex(tok.decode(seq.tolist(), skip_special=True))
            ref = normalize_tex(tok.decode(y[i].tolist(), skip_special=True))

            preds.append(prd)
            refs.append(ref)

            if printed < args.print_examples:
                print("REF:", ref)
                print("PRD:", prd)
                print("-" * 40)
                printed += 1

    bleu_char = corpus_char_bleu(preds, refs, max_n=4, smooth=1.0)
    em = exact_match(preds, refs)
    edit_sim = normalized_edit_similarity(preds, refs)

    print(f"Char-BLEU = {bleu_char:.3f}")
    print(f"ExactMatch = {em:.3f}%")
    print(f"EditSimilarity = {edit_sim:.3f}%")


if __name__ == "__main__":
    main()