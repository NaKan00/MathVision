import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_ids: [B, T]
        memory:  [S, B, D]
        return:  [B, T, V]
        """
        B, T = tgt_ids.shape
        tgt = self.emb(tgt_ids) * math.sqrt(self.d_model)
        tgt = self.emb_drop(tgt)
        tgt = tgt.transpose(0, 1)
        tgt = self.pos(tgt)

        tgt_mask = self.causal_mask(T, tgt.device)
        tgt_key_padding_mask = (tgt_ids == self.pad_id)

        h = self.dec(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.out(h).transpose(0, 1)
        return logits