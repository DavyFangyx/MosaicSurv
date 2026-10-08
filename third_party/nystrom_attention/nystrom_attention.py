# Vendored from https://github.com/lucidrains/nystrom-attention
# Copyright (c) 2021 Phil Wang (lucidrains), MIT License.
# 本地修改：forward 的 mask 路径（landmark 全为 padding 时的 landmark_mask 处理）。
import torch
import torch.nn as nn
import torch.nn.functional as F


def _moore_penrose_iter_pinv(x, iters=6):
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (col.max().clamp(min=1e-6) * row.max().clamp(min=1e-6))

    identity = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    while identity.dim() < x.dim():
        identity = identity.unsqueeze(0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * identity - xz @ (15 * identity - xz @ (7 * identity - xz)))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = heads
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.eps = eps
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                kernel_size=(residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    def forward(self, x, mask=None, return_attn=False, return_attn_matrices=False):
        batch_size, original_len, _ = x.shape
        num_landmarks = min(self.num_landmarks, original_len)
        pad_len = (num_landmarks - original_len % num_landmarks) % num_landmarks

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        seq_len = x.shape[1]
        segment_len = seq_len // num_landmarks

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.reshape(batch_size, seq_len, self.heads, -1).transpose(1, 2)
            for tensor in qkv
        ]

        if mask is not None:
            mask = mask.bool()
            valid = mask[:, None, :, None].to(dtype=q.dtype)
            q = q * valid
            k = k * valid
            v = v * valid

        q = q * self.scale

        q_landmarks = q.reshape(batch_size, self.heads, num_landmarks, segment_len, -1).sum(dim=3)
        k_landmarks = k.reshape(batch_size, self.heads, num_landmarks, segment_len, -1).sum(dim=3)

        landmark_mask = None
        if mask is None:
            divisor = float(segment_len)
        else:
            landmark_counts = mask.reshape(batch_size, num_landmarks, segment_len).sum(dim=-1)
            landmark_mask = landmark_counts > 0
            divisor = landmark_counts[:, None, :, None].clamp(min=self.eps).to(dtype=q.dtype)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        sim1 = q @ k_landmarks.transpose(-1, -2)
        sim2 = q_landmarks @ k_landmarks.transpose(-1, -2)
        sim3 = q_landmarks @ k.transpose(-1, -2)

        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            token_mask = mask[:, None, :]
            landmark_mask = landmark_mask[:, None, :]
            sim1 = sim1.masked_fill(~(token_mask[..., None] & landmark_mask[..., None, :]), mask_value)
            sim2 = sim2.masked_fill(~(landmark_mask[..., None] & landmark_mask[..., None, :]), mask_value)
            sim3 = sim3.masked_fill(~(landmark_mask[..., None] & token_mask[..., None, :]), mask_value)

        attn1 = sim1.softmax(dim=-1)
        attn2 = sim2.softmax(dim=-1)
        attn3 = sim3.softmax(dim=-1)
        attn2_inv = _moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out = out + self.res_conv(v)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        if mask is not None:
            out = out * mask[:, :, None].to(dtype=out.dtype)

        out = out[:, :original_len]

        if return_attn_matrices:
            return out, (attn1, attn2_inv, attn3)
        if return_attn:
            return out, attn1 @ attn2_inv @ attn3
        return out
