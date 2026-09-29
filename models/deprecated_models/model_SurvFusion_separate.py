import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力池化层：对变长序列（如 WSI patches）进行加权聚合
    支持两种输入格式：
    - 输入: (L, D) → 输出: (D,)        [单个样本的patches]
    - 输入: (B, L, D) → 输出: (B, D)   [批量样本的patches]
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
            attn_scores = self.attention(x)          # (L, 1)
            attn_weights = F.softmax(attn_scores, dim=0)
            return (x * attn_weights).sum(dim=0)     # (D,)
        elif x.dim() == 3:
            B, L, D = x.shape
            x_reshaped = x.reshape(B * L, D)
            attn_scores = self.attention(x_reshaped).reshape(B, L, 1)
            attn_weights = F.softmax(attn_scores, dim=1)
            return (x * attn_weights).sum(dim=1)     # (B, D)
        else:
            raise ValueError(f"AttentionPooling expects 2D or 3D input, got {x.dim()}D")


class SurvFusion(nn.Module):
    """
    两阶段分离训练的 SurvFusion 模型（WSI病理图像 I、基因组学 T、临床数据 S）

    Stage1 (CLIP + WSI池化)：
      - 训练 wsi_pooling、proj_*、norm_*、temperature
      - 损失：alignment_loss（对齐损失）
      - 验证标准：alignment_loss 最小

    Stage2 (多头注意力 + Classifier)：
      - 冻结 Stage1 参数，训练 modal_attention、classifier
      - 损失：survival_loss（生存损失）
      - 验证标准：c_index 最大

    架构：
      ⓪ WSI 注意力池化
      ① Feature Reprojection Module  —— 三路 Linear 映射到统一维度 D
      ② CLIP-based Multi-modal Alignment —— 三维两两对比损失（Stage1 使用）
      ③ 多头自注意力融合 —— 三模态作为等地位的 3-token 序列
      ④ Concat —— 拼接三个模态的注意力输出
      ⑤ Classifier —— Linear→LN→GELU→Dropout→Linear → 风险分数
    """

    def __init__(
        self,
        wsi_embedding_dim: int = 1024,
        gene_embedding_dim: int = 768,
        clinic_embedding_dim: int = 512,
        projection_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        training_stage: str = "stage1",  # "stage1" 或 "stage2"
        fusion_type: str = "mhsa",       # 实验3: "mhsa" | "concat" | "mean_concat"
        clip_weight_IT: float = 1.0,     # 实验4: I-T 对权重
        clip_weight_IS: float = 1.0,     # 实验4: I-S 对权重
        clip_weight_TS: float = 1.0,     # 实验4: T-S 对权重
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.training_stage = training_stage
        self.fusion_type = fusion_type
        self.clip_weight_IT = clip_weight_IT
        self.clip_weight_IS = clip_weight_IS
        self.clip_weight_TS = clip_weight_TS
        self.temperature = nn.Parameter(torch.tensor(temperature).log())

        # ⓪ 三模态注意力池化（将 token 序列聚合为单一向量）
        self.wsi_pooling    = AttentionPooling(wsi_embedding_dim)
        self.gene_pooling   = AttentionPooling(gene_embedding_dim)
        self.clinic_pooling = AttentionPooling(clinic_embedding_dim)

        # ① Feature Reprojection Module
        self.proj_I = nn.Linear(wsi_embedding_dim,    projection_dim)
        self.proj_T = nn.Linear(gene_embedding_dim,   projection_dim)
        self.proj_S = nn.Linear(clinic_embedding_dim, projection_dim)

        self.norm_I = nn.LayerNorm(projection_dim)
        self.norm_T = nn.LayerNorm(projection_dim)
        self.norm_S = nn.LayerNorm(projection_dim)

        # ③ 多头自注意力：三个模态作为长度为 3 的 token 序列，地位完全对等
        self.modal_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(projection_dim)

        # ⑤ Classifier（输入维度取决于 fusion_type）
        clf_input_dim = projection_dim if fusion_type == "mean_concat" else projection_dim * 3
        clf_hidden_dim = projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, clf_hidden_dim),
            nn.LayerNorm(clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def set_training_stage(self, stage: str):
        """切换训练阶段"""
        if stage not in ["stage1", "stage2"]:
            raise ValueError(f"training_stage 必须是 'stage1' 或 'stage2'，得到 {stage}")

        self.training_stage = stage

    def freeze_stage1_modules(self):
        """冻结 Stage1 的所有参数（三模态池化、重投影、对齐参数）"""
        for module in [self.wsi_pooling, self.gene_pooling, self.clinic_pooling,
                       self.proj_I, self.proj_T, self.proj_S,
                       self.norm_I, self.norm_T, self.norm_S]:
            for param in module.parameters():
                param.requires_grad = False
        self.temperature.requires_grad = False

    def unfreeze_stage1_modules(self):
        """解冻 Stage1 的所有参数"""
        for module in [self.wsi_pooling, self.gene_pooling, self.clinic_pooling,
                       self.proj_I, self.proj_T, self.proj_S,
                       self.norm_I, self.norm_T, self.norm_S]:
            for param in module.parameters():
                param.requires_grad = True
        self.temperature.requires_grad = True

    def unfreeze_stage2_modules(self):
        """解冻 Stage2 的所有参数（多头注意力、分类器）"""
        for module in [self.modal_attention, self.classifier]:
            for param in module.parameters():
                param.requires_grad = True

    @staticmethod
    def clip_contrastive_loss(
        feat_I: torch.Tensor,
        feat_T: torch.Tensor,
        feat_S: torch.Tensor,
        temperature: torch.Tensor,
        w_IT: float = 1.0,
        w_IS: float = 1.0,
        w_TS: float = 1.0,
    ) -> torch.Tensor:
        """
        CLIP 对比对齐损失：三模态两两对比，支持自定义权重（实验4）

        Args:
            feat_I: (B, D) WSI 特征（L2 normalize 后）
            feat_T: (B, D) 基因特征（L2 normalize 后）
            feat_S: (B, D) 临床特征（L2 normalize 后）
            temperature: 可学习的温度参数
            w_IT, w_IS, w_TS: 三对损失的权重（默认均为 1.0）

        Returns:
            alignment_loss: 标量损失
        """
        B = feat_I.size(0)
        if B == 1:
            return feat_I.sum() * 0.0

        labels = torch.arange(B, device=feat_I.device)
        temp = temperature.exp().clamp(min=1e-4)

        def pair_loss(a, b):
            logits = torch.matmul(a, b.mT) / temp
            return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.mT, labels)) / 2.0

        total_w = w_IT + w_IS + w_TS
        return (w_IT * pair_loss(feat_I, feat_T)
                + w_IS * pair_loss(feat_I, feat_S)
                + w_TS * pair_loss(feat_T, feat_S)) / total_w

    def forward(
        self,
        x_path: torch.Tensor,
        x_omic: torch.Tensor,
        x_clinic: torch.Tensor,
        return_align_loss: bool = True,
    ):
        """
        前向传播，根据 training_stage 返回不同的输出
        
        Stage1: 返回 (None, alignment_loss)  - 只关心对齐损失
        Stage2: 返回 (logits, None)          - 只关心生存预测
        
        Args:
            x_path: (B, L, D_wsi) WSI patches 或 (B, D_wsi)
            x_omic: (B, D_gene) 基因特征
            x_clinic: (B, D_clinic) 临床特征
            return_align_loss: 是否计算对齐损失（仅 Stage1 需要）
        
        Returns:
            logits: (B, num_classes) 或 None
            alignment_loss: 标量 或 None
        """
        # ── ⓪ 三模态注意力池化（token 序列 → 单向量）
        # WSI: (B, L, D_wsi) or (L, D_wsi) → (B, D_wsi)
        if x_path.dim() == 3:
            x_path = self.wsi_pooling(x_path)          # (B, D_wsi)
        elif x_path.dim() == 2:
            x_path = self.wsi_pooling(x_path).unsqueeze(0)  # (1, D_wsi)
        # dim==1: 已经是单向量，后面统一 unsqueeze

        # Gene: (B, n_tokens, D_gene) or (n_tokens, D_gene) → (B, D_gene)
        if x_omic.dim() == 3:
            x_omic = self.gene_pooling(x_omic)          # (B, D_gene)
        elif x_omic.dim() == 2:
            x_omic = self.gene_pooling(x_omic).unsqueeze(0)  # (1, D_gene)

        # Clinic: (B, n_tokens, D_clinic) or (n_tokens, D_clinic) → (B, D_clinic)
        if x_clinic.dim() == 3:
            x_clinic = self.clinic_pooling(x_clinic)         # (B, D_clinic)
        elif x_clinic.dim() == 2:
            x_clinic = self.clinic_pooling(x_clinic).unsqueeze(0)  # (1, D_clinic)

        # 统一保证 batch dim 存在
        if x_path.dim() == 1:
            x_path = x_path.unsqueeze(0)
        if x_omic.dim() == 1:
            x_omic = x_omic.unsqueeze(0)
        if x_clinic.dim() == 1:
            x_clinic = x_clinic.unsqueeze(0)

        # ── ① Feature Reprojection
        I_prime = self.norm_I(self.proj_I(x_path))    # (B, D)
        T_prime = self.norm_T(self.proj_T(x_omic))    # (B, D)
        S_prime = self.norm_S(self.proj_S(x_clinic))  # (B, D)

        # ── ② CLIP Alignment Loss（仅 Stage1 需要）
        alignment_loss = None
        if self.training_stage == "stage1" and return_align_loss:
            I_norm = F.normalize(I_prime, dim=-1)
            T_norm = F.normalize(T_prime, dim=-1)
            S_norm = F.normalize(S_prime, dim=-1)
            alignment_loss = self.clip_contrastive_loss(
                I_norm, T_norm, S_norm, self.temperature,
                w_IT=self.clip_weight_IT,
                w_IS=self.clip_weight_IS,
                w_TS=self.clip_weight_TS,
            )

        if self.training_stage == "stage1":
            return None, alignment_loss

        # ── ③ 融合（Stage2，fusion_type 控制消融策略）
        if self.fusion_type == "mhsa":
            # 多头自注意力：三模态作为 3-token 序列
            tokens = torch.stack([I_prime, T_prime, S_prime], dim=1)  # (B, 3, D)
            attn_out, _ = self.modal_attention(tokens, tokens, tokens)
            attn_out = self.attn_norm(attn_out + tokens)
            fused = torch.cat([attn_out[:, 0], attn_out[:, 1], attn_out[:, 2]], dim=-1)  # (B, 3D)

        elif self.fusion_type == "concat":
            # 直接拼接，无注意力
            fused = torch.cat([I_prime, T_prime, S_prime], dim=-1)  # (B, 3D)

        elif self.fusion_type == "mean_concat":
            # 三路特征均值池化后得到单一融合向量
            fused = (I_prime + T_prime + S_prime) / 3.0  # (B, D)

        else:
            raise ValueError(f"未知 fusion_type: {self.fusion_type}，应为 'mhsa'/'concat'/'mean_concat'")

        # ── ④ Classifier
        logits = self.classifier(fused)   # (B, num_classes)

        return logits, None
