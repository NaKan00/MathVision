from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from im2latex.data.tokenizer import Tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex
from im2latex.utils import save_tokenizer


def make_loader(csv_path, images_dir, tok, batch_size, shuffle, max_len, height, max_width):
    tf = build_image_transform(height=height, max_width=max_width)
    ds = Im2LatexDataset(csv_path, images_dir, tok, tf, max_len=max_len)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
    )
    return dl


@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in tqdm(dl, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)

        tgt_inp = y[:, :-1]
        tgt_out = y[:, 1:]

        logits = model(x, tgt_inp)
        B, T, V = logits.shape

        loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        nonpad = (tgt_out != loss_fn.ignore_index).sum().item()

        total_loss += loss.item() * nonpad
        total_tokens += nonpad

    return total_loss / max(total_tokens, 1)


def save_checkpoint(path: Path, model, opt, epoch, step, tok, d_model, encoder_variant):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "vocab_size": len(tok.vocab.itos),
        "pad_id": tok.vocab.pad,
        "d_model": d_model,
        "encoder_variant": encoder_variant,
    }
    torch.save(ckpt, path)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    TRAIN = "datasets/im2latex/train.csv"
    VAL = "datasets/im2latex/val.csv"
    IMAGES = "datasets/im2latex/images/formula_images_processed"

    OUT_DIR = Path("checkpoints/im2latex_convnext")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- параметры, которые стабильно работают на M4/16GB ----
    encoder_variant = "small"  # можно "tiny" если хочется быстрее/стабильнее
    d_model = 256
    batch_size = 4
    height = 64
    max_width = 384
    max_len = 256
    num_epochs = 5
    save_every_steps = 500

    # tokenizer по train
    df = pd.read_csv(TRAIN)
    tok = Tokenizer.build(df["formula"].astype(str).tolist(), min_freq=2, max_size=8000)
    save_tokenizer(tok, OUT_DIR / "tokenizer.json")

    train_dl = make_loader(TRAIN, IMAGES, tok, batch_size, True, max_len, height, max_width)
    val_dl = make_loader(VAL, IMAGES, tok, batch_size, False, max_len, height, max_width)

    model = Img2Latex(
        vocab_size=len(tok.vocab.itos),
        pad_id=tok.vocab.pad,
        d_model=d_model,
        encoder_variant=encoder_variant,
        encoder_pretrained=False,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tok.vocab.pad)

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train e{epoch}", leave=True)

        for x, y in pbar:
            global_step += 1
            x = x.to(device)
            y = y.to(device)

            tgt_inp = y[:, :-1]
            tgt_out = y[:, 1:]

            logits = model(x, tgt_inp)
            B, T, V = logits.shape

            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.3f}", step=global_step)

            if global_step % save_every_steps == 0:
                save_checkpoint(OUT_DIR / "last.pt", model, opt, epoch, global_step, tok, d_model, encoder_variant)

        val_loss = evaluate(model, val_dl, loss_fn, device)
        print(f"\nepoch {epoch}: val_loss={val_loss:.4f}")

        save_checkpoint(OUT_DIR / f"epoch_{epoch}.pt", model, opt, epoch, global_step, tok, d_model, encoder_variant)
        save_checkpoint(OUT_DIR / "last.pt", model, opt, epoch, global_step, tok, d_model, encoder_variant)


if __name__ == "__main__":
    main()