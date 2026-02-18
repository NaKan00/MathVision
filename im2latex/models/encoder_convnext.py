import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small


class ConvNeXtEncoder(nn.Module):
    """
    Input:  x [B, 1, 64, W]
    Output: memory [S, B, D]
    """

    def __init__(
        self,
        variant: str = "small",
        d_model: int = 256,
        pretrained: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        if variant not in {"tiny", "small"}:
            raise ValueError("variant must be 'tiny' or 'small'")

        # grayscale -> RGB
        self.in_proj = nn.Conv2d(1, 3, kernel_size=1)

        if variant == "tiny":
            m = convnext_tiny(weights="DEFAULT" if pretrained else None)
        else:
            m = convnext_small(weights="DEFAULT" if pretrained else None)

        self.features = m.features

        # determine channel size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 256)
            c_out = self.features(dummy).shape[1]

        # project to d_model
        self.out_proj = nn.Conv2d(c_out, d_model, kernel_size=1)

        # regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)          # [B,3,64,W]
        feat = self.features(x)      # [B,C,H',W']
        feat = self.out_proj(feat)   # [B,D,H',W']

        B, D, H, W = feat.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        feat = feat.view(B, H * W, D)                 # [B,S,D]

        feat = self.dropout(feat)
        feat = self.norm(feat)

        memory = feat.transpose(0, 1)  # [S,B,D]
        return memory