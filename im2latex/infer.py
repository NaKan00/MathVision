from pathlib import Path

import torch
from PIL import Image

from im2latex.models.seq2seq import Img2Latex
from im2latex.data.transforms import build_image_transform
from im2latex.utils import load_tokenizer


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
    x = tf(img).unsqueeze(0).to(device)

    ids = beam_decode_single(
        model,
        x,
        bos_id=tok.vocab.bos,
        eos_id=tok.vocab.eos,
        pad_id=tok.vocab.pad,
        beam_size=5,
        max_len=160,
        length_penalty_alpha=0.6,
        no_repeat_ngram=3,
    ).tolist()

    latex = tok.decode(ids, skip_special=True)
    print("pred:", latex)


if __name__ == "__main__":
    main()