
from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
import warnings

class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def masked_mean(x, padding_mask=None):
    if padding_mask is None:
        return x.mean(dim=1)

    valid_mask = (~padding_mask).unsqueeze(-1).float()
    denom = valid_mask.sum(dim=1).clamp(min=1.0)
    return (x * valid_mask).sum(dim=1) / denom


def modality_dropout(available_mask, drop_prob=0.2, training=False):
    if not training or drop_prob <= 0:
        return available_mask

    keep_mask = available_mask.clone()
    random_drop = torch.rand_like(keep_mask.float()) < drop_prob
    keep_mask = keep_mask & (~random_drop)

    empty_rows = keep_mask.sum(dim=1) == 0
    if empty_rows.any():
        fallback_indices = torch.argmax(available_mask[empty_rows].float(), dim=1)
        keep_mask[empty_rows] = False
        keep_mask[empty_rows, fallback_indices] = True

    return keep_mask


class VAETransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        attn_out, _ = self.attn(
            self.attn_norm(x),
            self.attn_norm(x),
            self.attn_norm(x),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NystromVAETransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8, mlp_dim=1024, num_landmarks=None, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // nhead,
            heads=nhead,
            num_landmarks=num_landmarks or dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        valid_mask = None if padding_mask is None else ~padding_mask
        x = x + self.attn(self.attn_norm(x), mask=valid_mask)
        x = x + self.ffn(self.ffn_norm(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class TokenSetEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, nhead=8, mlp_dim=1024, num_layers=1, dropout=0.1):
        super().__init__()
        self.mu_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.logvar_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [VAETransformerBlock(input_dim, nhead=nhead, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(input_dim)
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        mu_query = self.mu_query.expand(batch_size, -1, -1)
        logvar_query = self.logvar_query.expand(batch_size, -1, -1)
        h = torch.cat([mu_query, logvar_query, x], dim=1)

        for block in self.blocks:
            h = block(h, padding_mask=None)

        h = self.norm(h)
        mu = self.to_mu(h[:, 0])
        logvar = self.to_logvar(h[:, 1]).clamp(min=-4.0, max=2.0)
        return mu, logvar


GeneClinicEncoder = TokenSetEncoder


class ResamplerCrossAttentionBlock(nn.Module):
    def __init__(self, dim, nhead=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens, context_tokens, padding_mask=None):
        attn_out, _ = self.attn(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + attn_out
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        return query_tokens


class WSIMILResampler(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        token_dim=768,
        num_tokens=16,
        nhead=8,
        mlp_dim=1024,
        num_layers=2,
        dropout=0.1,
        max_tokens=4096,
        use_self_attention=True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.input_proj = nn.Linear(input_dim, token_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [ResamplerCrossAttentionBlock(token_dim, nhead=nhead, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.self_block = VAETransformerBlock(token_dim, nhead=nhead, mlp_dim=mlp_dim, dropout=dropout) if use_self_attention else None

    def forward(self, x, padding_mask=None):
        batch_size = x.shape[0]
        if x.shape[1] > self.max_tokens:
            keep_idx = torch.linspace(0, x.shape[1] - 1, self.max_tokens, device=x.device).long()
            x = x.index_select(dim=1, index=keep_idx)
            if padding_mask is not None:
                padding_mask = padding_mask.index_select(dim=1, index=keep_idx)

        tokens = self.input_proj(x)
        if padding_mask is not None and padding_mask.shape[1] != tokens.shape[1]:
            padding_mask = None
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        for block in self.cross_blocks:
            query_tokens = block(query_tokens, tokens, padding_mask=padding_mask)

        if self.self_block is not None:
            query_tokens = self.self_block(query_tokens, padding_mask=None)

        return query_tokens


class WSITargetPoolingHead(nn.Module):
    def __init__(self, token_dim=768, output_dim=768):
        super().__init__()
        self.proj = nn.Linear(token_dim, output_dim)

    def forward(self, tokens, padding_mask=None):
        pooled = tokens.mean(dim=1)
        return self.proj(pooled)


def parse_alphapgc(alphapgc):
    if alphapgc is None:
        raise ValueError("alphafix requires alphapgc weights for pathology,gene,clinic, e.g. 0.5,0.3,0.2")
    if isinstance(alphapgc, str):
        parts = [part.strip() for part in alphapgc.split(",")]
    else:
        parts = list(alphapgc)
    if len(parts) != 3:
        raise ValueError(
            f"alphapgc must have 3 comma-separated weights for pathology,gene,clinic, got `{alphapgc}`."
        )
    try:
        weights = [float(part) for part in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alphapgc must be 3 numeric weights for pathology,gene,clinic, got `{alphapgc}`."
        ) from exc
    if any(weight < 0 for weight in weights):
        raise ValueError(f"alphapgc weights must be non-negative, got `{alphapgc}`.")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"alphapgc weights must sum to a positive value, got `{alphapgc}`.")
    return [weight / total for weight in weights]


class GeneralizedPoE(nn.Module):
    def __init__(self, num_modalities=3, alphafix=False, alphapgc=None):
        super().__init__()
        self.num_modalities = num_modalities
        self.alphafix = bool(alphafix)
        if self.alphafix:
            weights = torch.tensor(parse_alphapgc(alphapgc), dtype=torch.float32)
            if weights.numel() != num_modalities:
                raise ValueError("alphapgc must match the number of PoE modalities.")
            self.register_buffer("fixed_alpha", weights)
        else:
            self.modality_logits = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, mus, logvars, available_mask):
        precision_terms = [torch.exp(-logvar.clamp(min=-4.0, max=2.0)) for logvar in logvars]
        stacked_mu = torch.stack(mus, dim=1)
        stacked_tau = torch.stack(precision_terms, dim=1)

        if self.alphafix:
            weights = self.fixed_alpha.to(dtype=stacked_mu.dtype, device=stacked_mu.device)
            weights = weights.unsqueeze(0).expand(available_mask.shape[0], -1)
            weights = torch.where(available_mask, weights, torch.zeros_like(weights))
            weight_sum = weights.sum(dim=1, keepdim=True)
            weights = torch.where(weight_sum > 0, weights / weight_sum.clamp_min(1e-12), weights)
        else:
            logits = self.modality_logits.unsqueeze(0).expand(available_mask.shape[0], -1)
            logits = logits.masked_fill(~available_mask, float("-inf"))
            weights = torch.softmax(logits, dim=1)
            weights = torch.where(available_mask, weights, torch.zeros_like(weights))

        weighted_tau = weights.unsqueeze(-1) * stacked_tau
        tau_joint = 1.0 + weighted_tau.sum(dim=1)
        mu_joint = (weighted_tau * stacked_mu).sum(dim=1) / tau_joint
        logvar_joint = torch.log(torch.reciprocal(tau_joint)).clamp(min=-4.0, max=2.0)
        return mu_joint, logvar_joint, weights


def reparameterize(mu, logvar, sample=True):
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class ReconstructionLoss(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.logvars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in dims
        })

    def forward(self, recon_dict, target_dict, available_mask=None):
        # available_mask is kept for call-site compatibility; reconstruction ignores it.
        losses = {}
        total = None
        for name in self.dims:
            recon = recon_dict[name]
            target = target_dict[name]
            count = int(recon.shape[0])
            losses[f"counts_{name}"] = count
            if count == 0:
                losses[name] = recon.new_tensor(float("nan"))
                continue

            recon_flat = recon.reshape(count, -1)
            target_flat = target.reshape(count, -1)
            mse = F.mse_loss(recon_flat, target_flat, reduction="none").mean(dim=1)
            logvar = self.logvars[name].clamp(min=-4.0, max=4.0)
            sigma2 = torch.exp(logvar)
            loss = mse / (2.0 * sigma2) + 0.5 * logvar
            loss = loss.mean()
            losses[name] = loss
            total = loss if total is None else total + loss

        if total is None:
            ref = next(iter(recon_dict.values()))
            total = ref.new_tensor(0.0)
        losses["total"] = total
        return losses


class JeffreysDivergence(nn.Module):
    def forward(self, mu, logvar):
        sigma2 = torch.exp(logvar.clamp(min=-4.0, max=2.0))
        inv_sigma2 = torch.reciprocal(sigma2)
        value = 0.5 * (sigma2 + inv_sigma2 - 2.0 + mu.pow(2) * (1.0 + inv_sigma2))
        return value.mean(dim=1).mean()


class KLDivergence(nn.Module):
    def forward(self, mu, logvar):
        logvar = logvar.clamp(min=-4.0, max=2.0)
        sigma2 = torch.exp(logvar)
        value = 0.5 * (mu.pow(2) + sigma2 - 1.0 - logvar)
        return value.mean(dim=1).mean()


PATTERNS = (1, 2, 3, 4, 5, 6, 7)
PATTERN_NAMES = {
    1: "P",
    2: "G",
    3: "PG",
    4: "C",
    5: "PC",
    6: "GC",
    7: "PGC",
}
MIN_BATCH_FOR_COX = 8


def as_pattern_ids(pattern_id, batch_size, device):
    if not torch.is_tensor(pattern_id):
        pattern_id = torch.as_tensor(pattern_id, device=device)
    pattern_id = pattern_id.to(device=device, dtype=torch.long)
    if pattern_id.ndim == 0:
        pattern_id = pattern_id.expand(int(batch_size))
    else:
        pattern_id = pattern_id.reshape(-1)
        if pattern_id.numel() == 1 and int(batch_size) > 1:
            pattern_id = pattern_id.reshape(()).expand(int(batch_size))
    pattern_id = pattern_id.reshape(-1)
    if int(pattern_id.numel()) != int(batch_size):
        raise ValueError(
            f"pattern_id has {pattern_id.numel()} values, expected batch_size={batch_size}."
        )
    return pattern_id


def pattern_to_mask(pattern_id, device=None, dtype=torch.bool):
    pid = int(pattern_id)
    bits = [(pid >> 0) & 1, (pid >> 1) & 1, (pid >> 2) & 1]
    return torch.tensor(bits, dtype=dtype, device=device)


def bits_to_id(avail_mask):
    bits = avail_mask.to(dtype=torch.long)
    return bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]


def _expand_avail(avail_mask, like):
    return avail_mask.to(device=like.device, dtype=torch.bool).view(-1, *([1] * (like.ndim - 1)))


def refill_missing_modalities(
    wsi_tokens,
    gene_tokens,
    clinic_tokens,
    recon_wsi,
    recon_gene,
    recon_clinic,
    available_mask,
):
    recon_wsi = recon_wsi.reshape(wsi_tokens.shape)
    recon_gene = recon_gene.reshape(gene_tokens.shape)
    recon_clinic = recon_clinic.reshape(clinic_tokens.shape)
    wsi_tokens_2 = torch.where(_expand_avail(available_mask[:, 0], wsi_tokens), wsi_tokens, recon_wsi)
    gene_tokens_2 = torch.where(_expand_avail(available_mask[:, 1], gene_tokens), gene_tokens, recon_gene)
    clinic_tokens_2 = torch.where(
        _expand_avail(available_mask[:, 2], clinic_tokens),
        clinic_tokens,
        recon_clinic,
    )
    return wsi_tokens_2, gene_tokens_2, clinic_tokens_2


class MultiPatternHead(nn.Module):
    def __init__(self, d_z=128, mmhid=256, dropout=0.1, label_dim=1):
        super().__init__()
        self.fuse_fc = nn.ModuleDict({
            str(s): nn.Sequential(
                nn.Linear(d_z, mmhid), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(dropout),
            ) for s in range(1, 8)
        })
        self.classifier = nn.Linear(mmhid, label_dim)

    def forward(self, mu_joint, pattern_id):
        pattern_id = as_pattern_ids(pattern_id, mu_joint.size(0), mu_joint.device)
        risk = mu_joint.new_zeros(mu_joint.size(0), self.classifier.out_features)
        for s in pattern_id.unique().tolist():
            sel = pattern_id == int(s)
            fused = self.fuse_fc[str(int(s))](mu_joint[sel])
            risk[sel] = self.classifier(fused)
        return risk


class FiLMHead(nn.Module):
    def __init__(self, d_z=128, mmhid=256, emb_dim=32, dropout=0.1, label_dim=1):
        super().__init__()
        self.emb = nn.Embedding(8, emb_dim)
        self.film = nn.Linear(emb_dim, 2 * d_z)
        self.fuse_fc = nn.Sequential(
            nn.Linear(d_z, mmhid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(mmhid, label_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, mu_joint, pattern_id):
        pattern_id = as_pattern_ids(pattern_id, mu_joint.size(0), mu_joint.device)
        e = self.emb(pattern_id)
        gamma, beta = self.film(e).chunk(2, dim=-1)
        z = (1.0 + gamma) * mu_joint + beta
        return self.classifier(self.fuse_fc(z))


def multi_pattern_surv_step(
    mu_list,
    logvar_list,
    avail_real,
    poe,
    head,
    event_time,
    censor,
    pattern_weights=None,
    min_batch_for_cox=MIN_BATCH_FOR_COX,
    loss_fn=None,
    stop_grad_poe=False,
):
    if loss_fn is None:
        from utils.loss_func import CoxSurvLoss
        loss_fn = CoxSurvLoss()
    if pattern_weights is None:
        pattern_weights = {s: 1.0 / float(len(PATTERNS)) for s in PATTERNS}

    device = mu_list[0].device
    batch_size = mu_list[0].shape[0]
    avail_real = avail_real.to(device=device, dtype=torch.bool)
    event_time = event_time.to(device=device)
    censor = censor.to(device=device)

    loss_surv = mu_list[0].new_tensor(0.0)
    risks = {}
    n_patterns = 0
    for s in PATTERNS:
        pm = pattern_to_mask(s, device=device)
        sel = (avail_real >= pm.unsqueeze(0)).all(dim=1)
        if int(sel.sum()) < int(min_batch_for_cox):
            continue
        mask_s = pm.unsqueeze(0).expand(batch_size, -1)[sel]
        if stop_grad_poe:
            # L_surv 不回传 Encoder / PoE alpha：联合后验只作为常量化输入给读出层
            with torch.no_grad():
                mu_s, logvar_s, _ = poe(
                    [mu[sel] for mu in mu_list],
                    [lv[sel] for lv in logvar_list],
                    mask_s,
                )
        else:
            mu_s, logvar_s, _ = poe(
                [mu[sel] for mu in mu_list],
                [lv[sel] for lv in logvar_list],
                mask_s,
            )
        risk_s = head(mu_s, pattern_id=s)
        loss_s = loss_fn(h=risk_s, t=event_time[sel], c=censor[sel])
        weight = float(pattern_weights[s])
        loss_surv = loss_surv + weight * loss_s
        risks[int(s)] = risk_s
        n_patterns += 1

    if n_patterns == 0:
        warnings.warn(
            "multi_pattern_surv_step skipped all 7 patterns because each subset had "
            f"fewer than {min_batch_for_cox} samples.",
            stacklevel=2,
        )
    return {
        "loss": loss_surv,
        "risks": risks,
        "n_patterns": n_patterns,
    }
