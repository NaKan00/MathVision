import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny, convnext_small


class ConvNeXtEncoder(nn.Module):
    

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

        
        self.repeat_gray_to_rgb = True

        
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
      
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat = self.features(x)     
        feat = self.out_proj(feat)  

        B, D, H, W = feat.shape

        feat = feat.permute(0, 2, 3, 1).contiguous()  
        feat = feat.view(B, H * W, D)                 
        feat = self.norm(feat)

        memory = feat.transpose(0, 1)                 

        memory_key_padding_mask = None

        if image_pad_mask is not None:
            
            pooled = F.interpolate(
                image_pad_mask.float().unsqueeze(1),
                size=W,
                mode="nearest",
            ).squeeze(1)

            pooled = pooled.bool()                    
            pooled = pooled.unsqueeze(1).expand(B, H, W)
            memory_key_padding_mask = pooled.reshape(B, H * W)

        return memory, memory_key_padding_mask