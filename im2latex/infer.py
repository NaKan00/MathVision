from pathlib import Path

import torch
from PIL import Image

from im2latex.models.seq2seq import Img2Latex
from im2latex.data.transforms import build_image_transform
from im2latex.utils import load_tokenizer


@torch.no_grad()
def greedy_decode(
    model: Img2Latex,
    image_tensor: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 256,
    no_repeat_ngram: int = 3,
):
    """
    Greedy decoding + анти-повторы (no_repeat_ngram).
    """
    model.eval()
    device = next(model.parameters()).device

    memory = model.encoder(image_tensor.to(device))
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    def block_repeated_ngrams(logits_row: torch.Tensor, seq: list[int], n: int):
        if n <= 0 or len(seq) < n:
            return logits_row
        prefix = seq[-(n - 1) :]
        banned = set()
        for i in range(len(seq) - n + 1):
            if seq[i : i + (n - 1)] == prefix:
                banned.add(seq[i + (n - 1)])
        if banned:
            idx = torch.tensor(list(banned), device=logits_row.device, dtype=torch.long)
            logits_row.index_fill_(0, idx, -1e9)
        return logits_row

    for _ in range(max_len):
        logits = model.decoder(ys, memory)
        next_logits = logits[0, -1].clone()

        next_logits[pad_id] = -1e9

        seq = ys[0].tolist()
        next_logits = block_repeated_ngrams(next_logits, seq, no_repeat_ngram)

        next_id = int(torch.argmax(next_logits, dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)

        if next_id == eos_id:
            break

    return ys[0].tolist()


@torch.no_grad()
def beam_search_decode(
    model: Img2Latex,
    image_tensor: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam: int = 5,
    max_len: int = 256,
    length_penalty: float = 0.7,
    no_repeat_ngram: int = 3,
):
    """
    Beam search на 1 картинку. Возвращает лучший список токенов.
    """
    model.eval()
    device = next(model.parameters()).device

    memory = model.encoder(image_tensor.to(device))

    def score_with_lp(logp: float, length: int) -> float:
        lp = ((5 + max(1, length)) / 6) ** length_penalty
        return logp / lp

    def block_repeated_ngrams(logits_row: torch.Tensor, seq: list[int], n: int):
        if n <= 0 or len(seq) < n:
            return logits_row
        prefix = seq[-(n - 1) :]
        banned = set()
        for i in range(len(seq) - n + 1):
            if seq[i : i + (n - 1)] == prefix:
                banned.add(seq[i + (n - 1)])
        if banned:
            idx = torch.tensor(list(banned), device=logits_row.device, dtype=torch.long)
            logits_row.index_fill_(0, idx, -1e9)
        return logits_row

    beams = [([bos_id], 0.0, False)]

    for _ in range(max_len):
        all_candidates = []

        for tokens, logp, finished in beams:
            if finished:
                all_candidates.append((tokens, logp, True))
                continue

            ys = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decoder(ys, memory)
            next_logits = logits[0, -1].clone()

            next_logits[pad_id] = -1e9
            next_logits = block_repeated_ngrams(next_logits, tokens, no_repeat_ngram)

            probs = torch.log_softmax(next_logits, dim=-1)
            topk = torch.topk(probs, k=beam)

            for next_id, add_logp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_tokens = tokens + [int(next_id)]
                new_logp = logp + float(add_logp)
                new_finished = (next_id == eos_id)
                all_candidates.append((new_tokens, new_logp, new_finished))

        all_candidates.sort(key=lambda x: score_with_lp(x[1], len(x[0])), reverse=True)
        beams = all_candidates[:beam]

        if all(b[2] for b in beams):
            break

    best_tokens, best_logp, _ = max(beams, key=lambda x: score_with_lp(x[1], len(x[0])))
    return best_tokens


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

    img_path = Path("datasets/im2latex/images/formula_images_processed") / "66667cee5b.png"

    tf = build_image_transform(height=64, max_width=384)
    img = Image.open(img_path)
    x = tf(img).unsqueeze(0)

    use_beam = False
    if use_beam:
        ids = beam_search_decode(
            model, x,
            bos_id=tok.vocab.bos, eos_id=tok.vocab.eos, pad_id=tok.vocab.pad,
            beam=5, max_len=256, length_penalty=0.7, no_repeat_ngram=3
        )
    else:
        ids = greedy_decode(
            model, x,
            bos_id=tok.vocab.bos, eos_id=tok.vocab.eos, pad_id=tok.vocab.pad,
            max_len=256, no_repeat_ngram=3
        )

    latex = tok.decode(ids, skip_special=True)
    print("pred:", latex)


if __name__ == "__main__":
    main()