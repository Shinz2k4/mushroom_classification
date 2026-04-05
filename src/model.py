import math
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── Squeeze-and-Excitation (SE) ──────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(1, channels // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


# ── ECANet ────────────────────────────────────────────────────────────────────
class ECABlock(nn.Module):
    def __init__(self, channels: int, b: int = 1, gamma: int = 2):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# ── MushroomNet ───────────────────────────────────────────────────────────────
class MushroomNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = tv_models.mobilenet_v3_large(
            weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # ── Đầu mạng ──────────────────────────────────────────────────────────
        self.conv_first = base.features[0]       # Conv2d 3→16, stride=2
        self.se1 = SEBlock(16)
        self.se2 = SEBlock(16)

        # ── Giữa và cuối features ─────────────────────────────────────────────
        self.middle   = base.features[1:-1]
        self.conv_last = base.features[-1]       # Conv2d 160→960

        # ── Sau avgpool: conv2d,1×1,NBN 960→1280 ───────────────────
  
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        _expand = nn.Conv2d(960, 1280, kernel_size=1, bias=True)
        _expand.weight.data = base.classifier[0].weight.data.view(1280, 960, 1, 1)
        _expand.bias.data   = base.classifier[0].bias.data
        self.expand = nn.Sequential(_expand, nn.Hardswish(inplace=True))

        # ── ECANet trên 1280-ch ───────────────────
        self.eca = ECABlock(1280)

        self.dropout = nn.Dropout(p=0.2)
        self.head    = nn.Linear(1280, num_classes)

    # ── Forward helpers ───────────────────────────────────────────────────────
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_first(x)
        x = self.se1(x)
        x = self.se2(x)
        x = self.middle(x)
        x = self.conv_last(x)          # (B, 960, 7, 7)
        x = self.avgpool(x)            # (B, 960, 1, 1)
        x = self.expand(x)             # (B, 1280, 1, 1)
        x = self.eca(x)                # (B, 1280, 1, 1) 
        return torch.flatten(x, 1)     # (B, 1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self._extract(x)))

    # ── Stage-3 freeze ────────────────────────────────────────────────────────
    def freeze_for_stage3(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in [self.conv_last, self.se1, self.se2, self.eca, self.head]:
            for p in m.parameters():
                p.requires_grad = True


# ── Factory functions ─────────────────────────────────────────────────────────
def build_mushroomnet(num_classes: int) -> MushroomNet:
    return MushroomNet(num_classes)


def build_resnet50(num_classes: int) -> nn.Module:
    """ResNet50 Transfer Learning"""
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
