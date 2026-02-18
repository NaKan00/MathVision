import torch
import torch.nn as nn

from .encoder_convnext import ConvNeXtEncoder
from .decoder import TransformerDecoder


class Img2Latex(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        encoder_variant: str = "small",
        encoder_pretrained: bool = False,  # было True
        dropout: float = 0.3,              # добавили (регуляризация)
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            variant=encoder_variant,
            d_model=d_model,
            pretrained=encoder_pretrained,
            dropout=0.1,                   # энкодеру достаточно слабого dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
            dropout=dropout,               # прокинули dropout декодера
        )

    def forward(self, images: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)          # [S,B,D]
        logits = self.decoder(tgt_inp, memory) # [B,T,V]
        return logits