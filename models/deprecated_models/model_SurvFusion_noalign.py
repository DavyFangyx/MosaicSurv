import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力池化层：对变长序列（如 WSI patches）进行加权聚合
    - 输入: (L, D) → 输出: (D,)
    - 输入: (B, L, D) → 输出: (B, D)
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            attn_scores = self.attention(x)
            attn_weights = F.softmax(attn_scores, dim=0)
            return (x * attn_weights).sum(dim=0)
        elif x.dim() == 3:
            B, L, D = x.shape
            x_reshaped = x.reshape(B * L, D)
            attn_scores = self.attention(x_reshaped).reshape(B, L, 1)
            attn_weights = F.softmax(attn_scores, dim=1)
            return (x * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"AttentionPooling expects 2D or 3D input, got {x.dim()}D")


class SurvFusionNoAlign(nn.Module):
    """
    对照实验1：去掉 CLIP，全部参数端到端只用 survival loss 训练。

    架构与 SurvFusion Stage2 相同，但无 Stage1/CLIP，
    所有参数（池化、重投影、注意力融合、分类器）从头端到端优化。

    建议 batch_size=32（--batch_size 32）。
    """

    def __init__(
        self,
        wsi_embedding_dim: int = 1024,
        gene_embedding_dim: int = 768,
        clinic_embedding_dim: int = 512,
        projection_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        # ⓪ 三模态注意力池化
        self.wsi_pooling    = AttentionPooling(wsi_embedding_dim)
        self.gene_pooling   = AttentionPooling(gene_embedding_dim)
        self.clinic_pooling = AttentionPooling(clinic_embedding_dim)

        # ① Feature Reprojection
        self.proj_I = nn.Linear(wsi_embedding_dim,    projection_dim)
        self.proj_T = nn.Linear(gene_embedding_dim,   projection_dim)
        self.proj_S = nn.Linear(clinic_embedding_dim, projection_dim)

        self.norm_I = nn.LayerNorm(projection_dim)
        self.norm_T = nn.LayerNorm(projection_dim)
        self.norm_S = nn.LayerNorm(projection_dim)

        # ② 多头自注意力融合
        self.modal_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(projection_dim)

        # ③ Classifier
        clf_input_dim = projection_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def forward(
        self,
        x_path: torch.Tensor,
        x_omic: torch.Tensor,
        x_clinic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_path:   (B, L, D_wsi) 或 (L, D_wsi)
            x_omic:   (B, D_gene)   或 (D_gene,)
            x_clinic: (B, D_clinic) 或 (D_clinic,)

        Returns:
            logits: (B, num_classes)
        """
        # ── ⓪ 注意力池化
        if x_path.dim() == 3:
            x_path = self.wsi_pooling(x_path)
        elif x_path.dim() == 2:
            x_path = self.wsi_pooling(x_path).unsqueeze(0)

        if x_omic.dim() == 3:
            x_omic = self.gene_pooling(x_omic)
        elif x_omic.dim() == 2:
            x_omic = self.gene_pooling(x_omic).unsqueeze(0)

        if x_clinic.dim() == 3:
            x_clinic = self.clinic_pooling(x_clinic)
        elif x_clinic.dim() == 2:
            x_clinic = self.clinic_pooling(x_clinic).unsqueeze(0)

        if x_path.dim() == 1:   x_path   = x_path.unsqueeze(0)
        if x_omic.dim() == 1:   x_omic   = x_omic.unsqueeze(0)
        if x_clinic.dim() == 1: x_clinic = x_clinic.unsqueeze(0)

        # ── ① Feature Reprojection
        I = self.norm_I(self.proj_I(x_path))    # (B, D)
        T = self.norm_T(self.proj_T(x_omic))    # (B, D)
        S = self.norm_S(self.proj_S(x_clinic))  # (B, D)

        # ── ② 多头自注意力融合
        tokens = torch.stack([I, T, S], dim=1)           # (B, 3, D)
        attn_out, _ = self.modal_attention(tokens, tokens, tokens)
        attn_out = self.attn_norm(attn_out + tokens)

        # ── ③ Concat + Classify
        fused = torch.cat([attn_out[:, 0], attn_out[:, 1], attn_out[:, 2]], dim=-1)
        return self.classifier(fused)
