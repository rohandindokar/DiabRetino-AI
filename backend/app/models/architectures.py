"""
Neural Network Architectures for Diabetic Retinopathy Severity Grading.
Combines Attention-Based CNNs (Squeeze-and-Excitation SEBlock, Spatial Attention)
and Vision Transformers (ViT) into a unified hybrid framework.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    Recalibrates channel-wise feature responses dynamically by explicitly
    modeling interdependencies between channels.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        se = self.global_avg_pool(x).view(batch_size, channels)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch_size, channels, 1, 1)
        return x * se


class SpatialAttention(nn.Module):
    """
    Spatial Attention Block.
    Computes spatial attention map using channel-wise average and max pooling
    convolved with a 7x7 filter to identify lesion hotspots.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return x * self.sigmoid(attn)


class SEResNet18(nn.Module):
    """
    Squeeze-and-Excitation ResNet18 for Diabetic Retinopathy Severity.
    Inserts channel attention recalibration after every residual stage.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super(SEResNet18, self).__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.se1 = SEBlock(in_channels=64)

        self.layer2 = base.layer2
        self.se2 = SEBlock(in_channels=128)

        self.layer3 = base.layer3
        self.se3 = SEBlock(in_channels=256)

        self.layer4 = base.layer4
        self.se4 = SEBlock(in_channels=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def get_last_conv_layer(self):
        return self.layer4

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = self.layer4(x)
        x = self.se4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        return self.fc(flattened)


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches (16x16) and projects to embedding dim.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 224, 224] -> [B, embed_dim, 14, 14] -> [B, embed_dim, 196] -> [B, 196, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerDR(nn.Module):
    """
    Vision Transformer (ViT) Classifier for Retinal Imaging.
    Models long-range global contextual relationships across retinal patches.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super(VisionTransformerDR, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        cls_out = features[:, 0]  # CLS token representation
        return self.head(cls_out)


class HybridViTAttentionCNN(nn.Module):
    """
    Flagship Hybrid Architecture combining:
    1. Attention-CNN (SE-ResNet18): captures fine-grained retinal textures & lesions.
    2. Vision Transformer (ViT): captures global spatial context & vascular distribution.
    3. Multi-modal Cross-Fusion Layer for multi-class DR severity classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super(HybridViTAttentionCNN, self).__init__()
        self.num_classes = num_classes

        # CNN Branch with SE Attention
        self.cnn_branch = SEResNet18(num_classes=num_classes, pretrained=pretrained)
        self.cnn_branch.fc = nn.Identity()  # Outputs 512-dim embedding

        # ViT Branch
        self.vit_branch = VisionTransformerDR(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=256,
            depth=6,
            num_heads=8
        )
        self.vit_branch.head = nn.Identity()  # Outputs 256-dim embedding

        # Fusion Classification Head (512 + 256 = 768)
        self.fusion_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def get_last_conv_layer(self):
        return self.cnn_branch.get_last_conv_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Branch
        cnn_feat = self.cnn_branch.forward_features(x)
        cnn_pooled = self.cnn_branch.avgpool(cnn_feat)
        cnn_vec = torch.flatten(cnn_pooled, 1)  # [B, 512]

        # ViT Branch
        vit_seq = self.vit_branch.forward_features(x)
        vit_vec = vit_seq[:, 0]  # [B, 256]

        # Fusion
        fused = torch.cat([cnn_vec, vit_vec], dim=1)  # [B, 768]
        gate = self.fusion_gate(fused)
        gated_fused = fused * gate

        logits = self.classifier(gated_fused)
        return logits
