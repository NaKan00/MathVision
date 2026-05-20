from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from im2latex.config import (
    TRAIN_CSV,
    IMAGES_DIR,
    IMAGE_HEIGHT,
    MAX_WIDTH,
    VOCAB_MIN_FREQ,
    VOCAB_MAX_SIZE,
    D_MODEL,
    ENCODER_VARIANT,
    MAX_LEN,
)
from im2latex.data.tokenizer import Tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex


SMALL_CSV = Path("overfit_small.csv")
OUT_CKPT = Path("checkpoints/overfit_small.pt")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    df = pd.read_csv(TRAIN_CSV).iloc[:100].copy()
    df.to_csv(SMALL_CSV, index=False)

    tok = Tokenizer.build(
        df["formula"].astype(str).tolist(),
        min_freq=1,
        max_size=VOCAB_MAX_SIZE,
    )

    tf = build_image_transform(
        height=IMAGE_HEIGHT,
        max_width=MAX_WIDTH,
    )

    ds = Im2LatexDataset(
        SMALL_CSV,
        IMAGES_DIR,
        tok,
        tf,
        max_len=MAX_LEN,
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
    )

    model = Img2Latex(
        vocab_size=len(tok.vocab.itos),
        pad_id=tok.vocab.pad,
        d_model=D_MODEL,
        encoder_variant=ENCODER_VARIANT,
        encoder_pretrained=True,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tok.vocab.pad,
    )

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(dl, desc=f"overfit e{epoch}", leave=True)

        for x, y, image_pad_mask in pbar:
            x = x.to(device)
            y = y.to(device)
            image_pad_mask = image_pad_mask.to(device)

            tgt_inp = y[:, :-1]
            tgt_out = y[:, 1:]

            logits = model(
                x,
                tgt_inp,
                image_pad_mask=image_pad_mask,
            )

            B, T, V = logits.shape

            loss = loss_fn(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            nonpad = (tgt_out != tok.vocab.pad).sum().item()
            total_loss += loss.item() * nonpad
            total_tokens += nonpad

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(total_tokens, 1)
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")

        if avg_loss < 0.15:
            print("Model can overfit small dataset. Architecture is probably OK.")
            break

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": len(tok.vocab.itos),
            "pad_id": tok.vocab.pad,
            "d_model": D_MODEL,
            "encoder_variant": ENCODER_VARIANT,
        },
        OUT_CKPT,
    )

    print("saved:", OUT_CKPT)


if __name__ == "__main__":
    main()