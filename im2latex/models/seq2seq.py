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
        encoder_pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            variant=encoder_variant,
            d_model=d_model,
            pretrained=encoder_pretrained,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
        )

    def forward(
        self,
        images: torch.Tensor,
        tgt_inp: torch.Tensor,
        image_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        memory, memory_key_padding_mask = self.encoder(
            images,
            image_pad_mask=image_pad_mask,
        )

        logits = self.decoder(
            tgt_inp,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return logits