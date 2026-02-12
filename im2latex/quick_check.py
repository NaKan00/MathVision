from im2latex.data.tokenizer import Tokenizer
from im2latex.data.transforms import build_image_transform
from im2latex.data.dataset import Im2LatexDataset, collate_batch

import pandas as pd
from torch.utils.data import DataLoader

TRAIN = "datasets/im2latex/train.csv"
IMAGES = "datasets/im2latex/images/formula_images_processed"

df = pd.read_csv(TRAIN)
tok = Tokenizer.build(df["formula"].astype(str).tolist(), min_freq=2, max_size=8000)

ds = Im2LatexDataset(TRAIN, IMAGES, tok, build_image_transform(64))
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_batch(b, tok.vocab.pad))

x, y = next(iter(dl))
print("x:", x.shape)  
print("y:", y.shape)   
print("sample decode:", tok.decode(y[0].tolist()))