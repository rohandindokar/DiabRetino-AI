"""
Neural Network Architectures for Diabetic Retinopathy Severity Grading.

Three models are provided, all exporting a 5-class softmax logit vector:

1. SE-ResNet18        - Squeeze-and-Excitation ResNet18 (attention CNN)
2. VisionTransformerDR- Vision Transformer (ViT) over 16x16 patches
3. HybridViTAttentionCNN- Flagship: SE-ResNet18 local features + ViT global
                          features fused through a gated multi-modal head.

All backbones accept (B, 3, 224, 224) tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# --------------------------------------------------------------------------- #
#  Squeeze-and-Excitation building block
# --------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    """Channel-recalibration block: global-pool -> reduce -> expand -> sigmoid."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, in_channels, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


# --------------------------------------------------------------------------- #
#  SE-ResNet18
# --------------------------------------------------------------------------- #
class SEResNet18(nn.Module):
    """ResNet18 with a SE attention block appended after each residual stage."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.bn1(self.relu(self.conv1(x))))
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.fc(torch.flatten(self.avgpool(feat), 1))

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer4


# --------------------------------------------------------------------------- #
#  Vision Transformer
# --------------------------------------------------------------------------- #
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, E, H, W)
        x = x.flatten(2).transpose(1, 2) # (B, n, E)
        return x


class VisionTransformerDR(nn.Module):
    """Compact ViT tuned to fit inside a 4GB GPU alongside the CNN branch."""

    def __init__(self, num_classes: int = 5, img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 192,
                 depth: int = 4, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, tokens), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def get_last_conv_layer(self) -> None:
        # No natural conv layer for Grad-CAM; caller falls back.
        return None


# --------------------------------------------------------------------------- #
#  Flagship hybrid: SE-ResNet18 + ViT with gated feature fusion
# --------------------------------------------------------------------------- #
class HybridViTAttentionCNN(nn.Module):
    """Fuses local (CNN) and global (ViT) features through a gated bottleneck."""

    CNN_DIM = 512
    VIT_DIM = 192

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        # CNN branch -> 512-d feature vector
        self.cnn = SEResNet18(num_classes=num_classes, pretrained=pretrained)
        self.cnn.fc = nn.Identity()  # keep 512-d embedding

        # ViT branch -> 192-d feature vector
        self.vit = VisionTransformerDR(num_classes=num_classes,
                                       embed_dim=self.VIT_DIM)
        self.vit.head = nn.Identity()  # keep 192-d CLS embedding

        fused_dim = self.CNN_DIM + self.VIT_DIM  # 704
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def features(self, x: torch.Tensor):
        cnn_vec = torch.flatten(self.cnn.avgpool(self.cnn.features(x)), 1)  # (B, 512)
        vit_vec = self.vit(torch.cat((
            self.vit.cls_token.expand(x.shape[0], -1, -1),
            self.vit.patch_embed(x),
        ), 1) + self.vit.pos_embed)  # run ViT trunk up to CLS embedding
        return cnn_vec, vit_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_vec = torch.flatten(self.cnn.avgpool(self.cnn.features(x)), 1)
        vit_vec = self.vit_head_features(x)
        fused = torch.cat([cnn_vec, vit_vec], dim=1)
        fused = fused * self.gate(fused)
        return self.classifier(fused)

    def vit_head_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.vit.patch_embed(x)
        seq = torch.cat((self.vit.cls_token.expand(b, -1, -1), tokens), dim=1)
        seq = seq + self.vit.pos_embed
        seq = self.vit.pos_drop(seq)
        seq = self.vit.transformer(seq)
        seq = self.vit.norm(seq)
        return seq[:, 0]  # CLS embedding (B, 192)

    def get_last_conv_layer(self) -> nn.Module:
        return self.cnn.get_last_conv_layer()


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
def build_model(name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    if name in ("hybrid_vit_cnn", "hybrid", "default"):
        return HybridViTAttentionCNN(num_classes=num_classes, pretrained=pretrained)
    if name == "se_resnet18":
        return SEResNet18(num_classes=num_classes, pretrained=pretrained)
    if name == "vit_dr":
        return VisionTransformerDR(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")

MODEL_REGISTRY = {
    "hybrid_vit_cnn": {
        "name": "Hybrid ViT + SE-Attention CNN (Flagship)",
        "description": "Gated fusion of SE-ResNet18 local lesion features and a Vision Transformer global-context branch.",
        "recommended": True,
    },
    "se_resnet18": {
        "name": "SE-ResNet18 (Attention CNN)",
        "description": "Squeeze-and-Excitation ResNet18 for local lesion texture detection.",
        "recommended": False,
    },
    "vit_dr": {
        "name": "Vision Transformer (ViT-Patch16)",
        "description": "Global self-attention transformer over 16x16 retinal patches.",
        "recommended": False,
    },
}
