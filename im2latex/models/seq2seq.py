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
        encoder_pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            variant=encoder_variant,
            d_model=d_model,
            pretrained=encoder_pretrained,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        logits = self.decoder(tgt_inp, memory)
        return logits