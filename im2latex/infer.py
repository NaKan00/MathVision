from pathlib import Path
import argparse

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
    max_len: int = 256,
):
    model.eval()
    device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)

    B, C, H, W = image_tensor.shape
    image_pad_mask = torch.zeros(B, W, dtype=torch.bool, device=device)

    memory, memory_key_padding_mask = model.encoder(
        image_tensor,
        image_pad_mask=image_pad_mask,
    )

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model.decoder(
            ys,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())

        ys = torch.cat(
            [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

        if next_id == eos_id:
            break

    return ys[0].tolist()


@torch.no_grad()
def beam_decode(
    model: Img2Latex,
    image_tensor: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 5,
    max_len: int = 256,
    repeat_penalty: float = 1.0,
):
    model.eval()
    device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)

    B, C, H, W = image_tensor.shape
    image_pad_mask = torch.zeros(B, W, dtype=torch.bool, device=device)

    memory, memory_key_padding_mask = model.encoder(
        image_tensor,
        image_pad_mask=image_pad_mask,
    )

    beams = [
        (
            torch.tensor([[bos_id]], dtype=torch.long, device=device),
            0.0,
            False,
        )
    ]

    for _ in range(max_len):
        candidates = []

        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue

            logits = model.decoder(
                seq,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            if repeat_penalty and repeat_penalty > 1.0:
                for prev_id in seq[0].tolist():
                    if prev_id not in {bos_id, eos_id}:
                        log_probs[prev_id] /= repeat_penalty

            top_scores, top_ids = torch.topk(log_probs, beam_size)

            for token_score, token_id in zip(top_scores, top_ids):
                token_id_int = int(token_id.item())

                new_seq = torch.cat(
                    [
                        seq,
                        torch.tensor(
                            [[token_id_int]],
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=1,
                )

                new_score = score + float(token_score.item())
                new_finished = token_id_int == eos_id

                candidates.append((new_seq, new_score, new_finished))

        candidates.sort(key=lambda x: x[1] / x[0].shape[1], reverse=True)
        beams = candidates[:beam_size]

        if all(finished for _, _, finished in beams):
            break

    best_seq = beams[0][0]
    return best_seq[0].tolist()


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
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default="datasets/im2latex/images/formula_images_processed/66667cee5b.png",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/im2latex_convnext/last.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/im2latex_convnext/tokenizer.json",
    )
    parser.add_argument(
        "--decode",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
    )

    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--repeat_penalty", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max_width", type=int, default=384)

    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = load_tokenizer(args.tokenizer)
    model = load_model(args.ckpt, device=device)

    tf = build_image_transform(
        height=args.height,
        max_width=args.max_width,
    )

    img = Image.open(args.image)
    x = tf(img).unsqueeze(0)

    if args.decode == "greedy":
        ids = greedy_decode(
            model,
            x,
            tok.vocab.bos,
            tok.vocab.eos,
            max_len=args.max_len,
        )
    else:
        ids = beam_decode(
            model,
            x,
            tok.vocab.bos,
            tok.vocab.eos,
            beam_size=args.beam_size,
            max_len=args.max_len,
            repeat_penalty=args.repeat_penalty,
        )

    latex = tok.decode(ids, skip_special=True)

    print("image:", args.image)
    print("decode:", args.decode)
    print("pred:", latex)


if __name__ == "__main__":
    main()