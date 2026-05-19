from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from .tokenizer import Tokenizer


class Im2LatexDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        tokenizer: Tokenizer,
        image_tf,
        max_len: int = 256,
    ):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_tf = image_tf
        self.max_len = max_len

        df = pd.read_csv(self.csv_path)
        if "image" not in df.columns or "formula" not in df.columns:
            raise ValueError(f"CSV must have columns [image, formula], got: {df.columns.tolist()}")

        self.rows = df[["image", "formula"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        img_path = self.images_dir / str(row["image"]).strip()
        formula = str(row["formula"])

        img = Image.open(img_path)
        x = self.image_tf(img)

        ids = self.tokenizer.encode(formula)
        y = torch.tensor(ids, dtype=torch.long)

        if y.shape[0] > self.max_len:
            y = y[: self.max_len].clone()
            y[-1] = self.tokenizer.vocab.eos

        return {"image": x, "tokens": y}


def collate_batch(batch: List[Dict], pad_id: int):
    images = [b["image"] for b in batch]  # [1,H,W]

    H = images[0].shape[1]
    widths = [im.shape[2] for im in images]
    maxW = max(widths)

    # image tensor
    x = torch.zeros(len(images), 1, H, maxW, dtype=images[0].dtype)

    # True = padding
    image_pad_mask = torch.ones(len(images), maxW, dtype=torch.bool)

    for i, im in enumerate(images):
        w = im.shape[2]

        x[i, :, :, :w] = im
        image_pad_mask[i, :w] = False

    # token padding
    toks = [b["tokens"] for b in batch]
    maxL = max(t.shape[0] for t in toks)

    y = torch.full((len(toks), maxL), pad_id, dtype=torch.long)

    for i, t in enumerate(toks):
        y[i, : t.shape[0]] = t

    return x, y, image_pad_mask