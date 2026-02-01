from pathlib import Path

import torch
from PIL import Image

from im2latex.models.seq2seq import Img2Latex
from im2latex.data.transforms import build_image_transform
from im2latex.utils import load_tokenizer


@torch.no_grad()
def greedy_decode(model: Img2Latex, image_tensor: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 256):
    model.eval()
    device = next(model.parameters()).device

    memory = model.encoder(image_tensor.to(device))  # [S,1,D]
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # [1,1]

    for _ in range(max_len):
        logits = model.decoder(ys, memory)  # [1,T,V]
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break

    return ys[0].tolist()


def load_model(ckpt_path: str | Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Img2Latex(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
        d_model=ckpt["d_model"],
        encoder_variant=ckpt.get("encoder_variant", "small"),
        encoder_pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/im2latex_convnext")

    tok = load_tokenizer(ckpt_dir / "tokenizer.json")
    model = load_model(ckpt_dir / "last.pt", device=device)

    # Пример: возьми любой png из датасета
    img_path = Path("datasets/im2latex/images/formula_images_processed") / "66667cee5b.png"

    tf = build_image_transform(height=64, max_width=384)
    img = Image.open(img_path)
    x = tf(img).unsqueeze(0)  # [1,1,64,W]

    ids = greedy_decode(model, x, tok.vocab.bos, tok.vocab.eos, max_len=256)
    latex = tok.decode(ids, skip_special=True)

    print("pred:", latex)


if __name__ == "__main__":
    main()