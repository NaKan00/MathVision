# im2latex/bleu_eval.py
# Запуск:
#   python -m im2latex.bleu_eval --ckpt checkpoints/im2latex_convnext/epoch_2.pt --max_samples 1000
#
# Важно: мы отключаем encoder_pretrained, чтобы НЕ пытаться скачивать convnext веса по сети.

import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
import sacrebleu

from im2latex.data.tokenizer import Tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex


def normalize_tex(s: str) -> str:
    return " ".join(str(s).strip().split())


@torch.no_grad()
def greedy_decode(model: Img2Latex, x: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 160):
    """
    x: [B, 1, H, W]
    return: tokens [B, T]
    """
    model.eval()
    B = x.size(0)
    device = x.device

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model(x, ys)  # [B, T, V]
        next_token = logits[:, -1, :].argmax(dim=-1)  # [B]
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        finished |= (next_token == eos_id)
        if bool(finished.all()):
            break

    return ys


def load_checkpoint_state(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")

    # Частые форматы:
    # 1) state_dict напрямую
    # 2) {"model": state_dict, ...}
    # 3) {"state_dict": state_dict, ...}
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # похоже на state_dict напрямую
        if all(isinstance(k, str) for k in obj.keys()):
            return obj

    raise ValueError(f"Не понял формат чекпоинта: {type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Путь к .pt (epoch_*.pt или last.pt)")
    ap.add_argument("--train_csv", default="datasets/im2latex/train.csv", help="train.csv для построения токенизатора")
    ap.add_argument("--val_csv", default="datasets/im2latex/val.csv", help="val.csv для оценки BLEU")
    ap.add_argument(
        "--images_dir",
        default="datasets/im2latex/images/formula_images_processed",
        help="Папка с png (где лежат картинки формул)",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_samples", type=int, default=1000, help="0 = все примеры")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=8000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--encoder_variant", default="small", help="convnext variant (small/base/large) если поддерживается")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    if not Path(args.train_csv).exists():
        raise FileNotFoundError(f"train_csv не найден: {args.train_csv}")
    if not Path(args.val_csv).exists():
        raise FileNotFoundError(f"val_csv не найден: {args.val_csv}")
    if not Path(args.images_dir).exists():
        raise FileNotFoundError(f"images_dir не найден: {args.images_dir}")

    # 1) Токенизатор строим по train.csv (как в обучении)
    train_df = pd.read_csv(args.train_csv)
    tok = Tokenizer.build(
        train_df["formula"].astype(str).tolist(),
        min_freq=args.min_freq,
        max_size=args.max_vocab,
    )

    # 2) Готовим val.csv (можно ограничить примеры)
    val_df = pd.read_csv(args.val_csv)
    tmp_csv = None
    if args.max_samples and args.max_samples > 0:
        val_df = val_df.iloc[: args.max_samples].copy()
        tmp_csv = Path(".bleu_tmp_val.csv")
        val_df.to_csv(tmp_csv, index=False)
        val_csv_path = str(tmp_csv)
    else:
        val_csv_path = args.val_csv

    ds = Im2LatexDataset(val_csv_path, args.images_dir, tok, build_image_transform(height=64))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
    )

    # 3) Модель: ОЧЕНЬ ВАЖНО — encoder_pretrained=False (чтобы не качать веса и не ловить SSL)
    # Параметры encoder_variant/encoder_pretrained должны совпадать с вашим Img2Latex.
    model = Img2Latex(
        vocab_size=len(tok.vocab.itos),
        pad_id=tok.vocab.pad,
        d_model=args.d_model,
        encoder_variant=args.encoder_variant,
        encoder_pretrained=False,
    ).to(device)

    state = load_checkpoint_state(str(ckpt_path))
    model.load_state_dict(state, strict=False)

    preds, refs = [], []

    for x, y in tqdm(dl, desc="BLEU eval"):
        x = x.to(device)
        y = y.to(device)

        gen = greedy_decode(
            model,
            x,
            bos_id=tok.vocab.bos,
            eos_id=tok.vocab.eos,
            max_len=args.max_len,
        )

        for gen_ids, tgt_ids in zip(gen.tolist(), y.tolist()):
            pred = tok.decode(gen_ids)
            ref = tok.decode(tgt_ids)
            preds.append(normalize_tex(pred))
            refs.append(normalize_tex(ref))

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    print(f"BLEU = {bleu:.2f}")

    if tmp_csv is not None and tmp_csv.exists():
        tmp_csv.unlink()


if __name__ == "__main__":
    main()