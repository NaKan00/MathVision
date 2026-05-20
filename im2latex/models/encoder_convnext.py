import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny, convnext_small


class ConvNeXtEncoder(nn.Module):
    """
    Input:
        x: [B, 1, 64, W]

    Output:
        memory: [S, B, D]
        memory_key_padding_mask: [B, S]

    OCR-friendly version:
    - не берём самый последний ConvNeXt stage;
    - сохраняем больше spatial tokens;
    - для формул это лучше, чем слишком сильное сжатие.
    """

    def __init__(
        self,
        variant: str = "small",
        d_model: int = 256,
        pretrained: bool = True,
        keep_until: int = 6,
    ):
        super().__init__()

        if variant not in {"tiny", "small"}:
            raise ValueError("variant must be 'tiny' or 'small'")

        if variant == "tiny":
            m = convnext_tiny(weights="DEFAULT" if pretrained else None)
        else:
            m = convnext_small(weights="DEFAULT" if pretrained else None)

        # ConvNeXt expects RGB.
        # Вместо обучаемого Conv2d(1 -> 3) просто повторяем grayscale в 3 канала.
        # Это стабильнее для pretrained ImageNet backbone.
        self.repeat_gray_to_rgb = True

        # Берём не весь backbone.
        # Полный ConvNeXt слишком сильно сжимает 64px height до ~2px.
        # keep_until=6 обычно даёт больше spatial resolution.
        self.features = nn.Sequential(
            *list(m.features.children())[:keep_until]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 512)
            feat = self.features(dummy)
            c_out = feat.shape[1]

            print(
                f"[ConvNeXtEncoder] variant={variant}, "
                f"keep_until={keep_until}, "
                f"feature_shape={tuple(feat.shape)}, "
                f"c_out={c_out}"
            )

        self.out_proj = nn.Conv2d(
            c_out,
            d_model,
            kernel_size=1,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        image_pad_mask: torch.Tensor | None = None,
    ):
        # x: [B,1,H,W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat = self.features(x)      # [B,C,H',W']
        feat = self.out_proj(feat)   # [B,D,H',W']

        B, D, H, W = feat.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        feat = feat.view(B, H * W, D)                 # [B,S,D]
        feat = self.norm(feat)

        memory = feat.transpose(0, 1)                 # [S,B,D]

        memory_key_padding_mask = None

        if image_pad_mask is not None:
            # image_pad_mask: [B, original_W], True = padding
            pooled = F.interpolate(
                image_pad_mask.float().unsqueeze(1),
                size=W,
                mode="nearest",
            ).squeeze(1)

            pooled = pooled.bool()                    # [B,W']
            pooled = pooled.unsqueeze(1).expand(B, H, W)
            memory_key_padding_mask = pooled.reshape(B, H * W)

        return memory, memory_key_padding_mask