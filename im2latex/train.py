from pathlib import Path
import random
import copy

import torch
from torch.utils.data import DataLoader, Sampler
import pandas as pd
from tqdm import tqdm

from im2latex.config import (
    TRAIN_CSV,
    VAL_CSV,
    IMAGES_DIR,
    CHECKPOINT_DIR,
    TOKENIZER_PATH,
    LAST_CKPT,
    BEST_CKPT,
    BEST_EMA_CKPT,
    IMAGE_HEIGHT,
    MAX_WIDTH,
    VOCAB_MIN_FREQ,
    VOCAB_MAX_SIZE,
    D_MODEL,
    ENCODER_VARIANT,
    ENCODER_PRETRAINED,
    BATCH_SIZE,
    MAX_LEN,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    SAVE_EVERY_STEPS,
    GRAD_ACCUM_STEPS,
    USE_LENGTH_BUCKETING,
    BUCKET_SIZE,
    USE_EMA,
    EMA_DECAY,
)

from im2latex.data.tokenizer import Tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch
from im2latex.models.seq2seq import Img2Latex
from im2latex.utils import save_tokenizer


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        self.ema.load_state_dict(state)


class LengthBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, bucket_size: int = 512, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        self.lengths = [
            len(str(row["formula"]))
            for row in dataset.rows
        ]

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        buckets = [
            indices[i: i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        batches = []

        for bucket in buckets:
            bucket.sort(key=lambda idx: self.lengths[idx])

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i: i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return len(self.dataset) // self.batch_size


def make_loader(csv_path, images_dir, tok, batch_size, shuffle):
    tf = build_image_transform(
        height=IMAGE_HEIGHT,
        max_width=MAX_WIDTH,
    )

    ds = Im2LatexDataset(
        csv_path,
        images_dir,
        tok,
        tf,
        max_len=MAX_LEN,
    )

    if shuffle and USE_LENGTH_BUCKETING:
        batch_sampler = LengthBucketBatchSampler(
            ds,
            batch_size=batch_size,
            bucket_size=BUCKET_SIZE,
            shuffle=True,
        )

        dl = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=lambda b: collate_batch(b, tok.vocab.pad),
        )
    else:
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

    for x, y, image_pad_mask in tqdm(dl, desc="val", leave=False):
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

        nonpad = (tgt_out != loss_fn.ignore_index).sum().item()

        total_loss += loss.item() * nonpad
        total_tokens += nonpad

    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    path: Path,
    model,
    opt,
    scheduler,
    epoch,
    step,
    tok,
    best_val_loss,
    ema=None,
    best_ema_val_loss=None,
):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "vocab_size": len(tok.vocab.itos),
        "pad_id": tok.vocab.pad,
        "d_model": D_MODEL,
        "encoder_variant": ENCODER_VARIANT,
        "encoder_pretrained": ENCODER_PRETRAINED,
        "best_val_loss": best_val_loss,
        "best_ema_val_loss": best_ema_val_loss,
    }

    if ema is not None:
        ckpt["ema_state"] = ema.state_dict()
        ckpt["ema_decay"] = ema.decay

    torch.save(ckpt, path)


def save_model_only_checkpoint(
    path: Path,
    model_state,
    epoch,
    step,
    tok,
    val_loss,
    is_ema: bool,
):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state": model_state,
        "vocab_size": len(tok.vocab.itos),
        "pad_id": tok.vocab.pad,
        "d_model": D_MODEL,
        "encoder_variant": ENCODER_VARIANT,
        "encoder_pretrained": ENCODER_PRETRAINED,
        "best_val_loss": val_loss,
        "is_ema": is_ema,
    }

    torch.save(ckpt, path)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    df["formula"] = df["formula"].astype(str)

    tok = Tokenizer.build(
        df["formula"].tolist(),
        min_freq=VOCAB_MIN_FREQ,
        max_size=VOCAB_MAX_SIZE,
    )

    save_tokenizer(tok, TOKENIZER_PATH)

    train_dl = make_loader(
        TRAIN_CSV,
        IMAGES_DIR,
        tok,
        BATCH_SIZE,
        True,
    )

    val_dl = make_loader(
        VAL_CSV,
        IMAGES_DIR,
        tok,
        BATCH_SIZE,
        False,
    )

    model = Img2Latex(
        vocab_size=len(tok.vocab.itos),
        pad_id=tok.vocab.pad,
        d_model=D_MODEL,
        encoder_variant=ENCODER_VARIANT,
        encoder_pretrained=ENCODER_PRETRAINED,
    ).to(device)

    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"EMA enabled: decay={EMA_DECAY}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=NUM_EPOCHS,
        eta_min=1e-5,
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tok.vocab.pad,
        label_smoothing=LABEL_SMOOTHING,
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_ema_val_loss = float("inf")

    if LAST_CKPT.exists():
        ckpt = torch.load(LAST_CKPT, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])

        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        if ema is not None and ckpt.get("ema_state") is not None:
            ema.load_state_dict(ckpt["ema_state"])
            print("EMA state restored from checkpoint.")

        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = last_epoch + 1
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_ema_val_loss = float(ckpt.get("best_ema_val_loss", float("inf")))

        print(
            f"Resuming from {LAST_CKPT}: "
            f"last_epoch={last_epoch}, "
            f"next_epoch={start_epoch}, "
            f"step={global_step}, "
            f"best_val_loss={best_val_loss:.4f}, "
            f"best_ema_val_loss={best_ema_val_loss:.4f}"
        )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        opt.zero_grad()

        pbar = tqdm(
            train_dl,
            desc=f"train e{epoch}",
            leave=True,
        )

        for x, y, image_pad_mask in pbar:
            global_step += 1

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

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if global_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                opt.step()

                if ema is not None:
                    ema.update(model)

                opt.zero_grad()

            current_lr = opt.param_groups[0]["lr"]

            pbar.set_postfix(
                loss=f"{loss.item() * GRAD_ACCUM_STEPS:.3f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
            )

            if global_step % SAVE_EVERY_STEPS == 0:
                save_checkpoint(
                    LAST_CKPT,
                    model,
                    opt,
                    scheduler,
                    epoch,
                    global_step,
                    tok,
                    best_val_loss,
                    ema=ema,
                    best_ema_val_loss=best_ema_val_loss,
                )

        val_loss = evaluate(
            model,
            val_dl,
            loss_fn,
            device,
        )

        print(f"\nepoch {epoch}: val_loss={val_loss:.4f}")

        ema_val_loss = None
        if ema is not None:
            ema_val_loss = evaluate(
                ema.ema,
                val_dl,
                loss_fn,
                device,
            )

            print(f"epoch {epoch}: ema_val_loss={ema_val_loss:.4f}")

        scheduler.step()

        save_checkpoint(
            CHECKPOINT_DIR / f"epoch_{epoch}.pt",
            model,
            opt,
            scheduler,
            epoch,
            global_step,
            tok,
            best_val_loss,
            ema=ema,
            best_ema_val_loss=best_ema_val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                BEST_CKPT,
                model,
                opt,
                scheduler,
                epoch,
                global_step,
                tok,
                best_val_loss,
                ema=ema,
                best_ema_val_loss=best_ema_val_loss,
            )

            print(
                f"new best checkpoint saved: "
                f"best.pt, val_loss={best_val_loss:.4f}"
            )

        if ema is not None and ema_val_loss is not None:
            if ema_val_loss < best_ema_val_loss:
                best_ema_val_loss = ema_val_loss

                save_model_only_checkpoint(
                    BEST_EMA_CKPT,
                    ema.state_dict(),
                    epoch,
                    global_step,
                    tok,
                    ema_val_loss,
                    is_ema=True,
                )

                print(
                    f"new best EMA checkpoint saved: "
                    f"best_ema.pt, ema_val_loss={best_ema_val_loss:.4f}"
                )

        save_checkpoint(
            LAST_CKPT,
            model,
            opt,
            scheduler,
            epoch,
            global_step,
            tok,
            best_val_loss,
            ema=ema,
            best_ema_val_loss=best_ema_val_loss,
        )


if __name__ == "__main__":
    main()