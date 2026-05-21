import math

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1536,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 512,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.pos_emb = nn.Embedding(
            max_len,
            d_model,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.dec = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # ВАЖНО: без weight tying.
        # Tied weights тут давали огромный loss и ломали overfit-test.
        self.out = nn.Linear(
            d_model,
            vocab_size,
        )

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                T,
                T,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = tgt_ids.shape

        if T > self.max_len:
            raise ValueError(
                f"Target length {T} exceeds decoder max_len={self.max_len}"
            )

        positions = torch.arange(
            T,
            device=tgt_ids.device,
        ).unsqueeze(0).expand(B, T)

        tgt = self.emb(tgt_ids) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_emb(positions)

        tgt_key_padding_mask = tgt_ids == self.pad_id
        tgt_mask = self.causal_mask(T, tgt.device)

        # Encoder memory сейчас приходит как [S, B, D],
        # а batch_first=True decoder ждёт [B, S, D].
        if memory.dim() == 3 and memory.shape[1] == B:
            memory = memory.transpose(0, 1)

        h = self.dec(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        h = self.norm(h)

        logits = self.out(h)

        return logits