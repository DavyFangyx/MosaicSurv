"""验证 survival_loss_from_cached 的 n_patterns==0 兜底修复。

复现条件：epoch 末班 batch 样本数 < min_batch_for_cox(=8) 时，
multi_pattern_surv_step 全部 7 个 pattern 子集都被跳过，loss 为无梯度常数 0。
A/B 变体（mosaic_surv_frozen / mosaic_surv_twostage）的 combine_loss 只含 survival 项，
backward 会抛 RuntimeError: element 0 of tensors does not require grad。
"""
import torch

from models.ablation_models.model_mosaic_surv_frozen import MosaicSurvFrozen
from models.ablation_models.model_mosaic_surv_twostage import MosaicSurvTwostage
from models.ablation_models.model_mosaic_surv import MosaicSurv


def fake_loss(h, t, c):
    return (h.view(-1) * t.view(-1)).mean()


def make_kwargs(variant):
    return dict(
        clinic_num_tokens=2,
        wsi_embedding_dim=64,
        gene_embedding_dim=64,
        clinic_embedding_dim=32,
        gene_num_tokens=4,
        latent_dim=128,
        mmhid=128,
        label_dim=1,
        decoder_hidden_dim=128,
        poe_variant=variant,
        selected_modalities="wsi,gene,clinic",
        poe_surv_lambda=1.0,
        modality_dropout_prob=0.35,
    )


def run_batch(model, B):
    x_path = torch.randn(B, 8, 64)
    x_omic = torch.randn(B, 4 * 64)
    x_clinic = torch.randn(B, 2 * 32)
    event_time = torch.rand(B, 1) * 100 + 1
    censor = torch.randint(0, 2, (B, 1)).float()
    avail = {k: torch.ones(B, dtype=torch.bool) for k in ("wsi", "gene", "clinic")}
    model(x_path=x_path, x_omic=x_omic, x_clinic=x_clinic, wsi_mask=None, avail=avail)
    loss = model.survival_loss_from_cached(event_time, censor, fake_loss)
    return loss


# --- Test 1: 小 batch（<8）触发兜底，A/B 变体 loss 有梯度，backward 不崩 ---
print("=== Test 1: small batch (<8) fallback keeps gradient ===")
for cls, name in ((MosaicSurvFrozen, "AFiLM(frozen)"), (MosaicSurvTwostage, "BFiLM(twostage)")):
    model = cls(**make_kwargs("A" if name.startswith("A") else "B"))
    # 模拟 stage2：冻结 backbone、只训 FiLM 头
    if name.startswith("A"):
        model.freeze_backbone_for_probe()
    else:
        model.freeze_backbone_keep_head()
    model.train()
    loss = run_batch(model, B=6)
    assert loss.requires_grad, f"{name}: small-batch loss 无梯度（兜底未生效）"
    model.zero_grad()
    loss.backward()
    head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pattern_head.parameters())
    assert head_grad, f"{name}: FiLM 头未收到梯度"
    print(f"OK: {name} small-batch loss requires_grad=True, FiLM head got grad")

# --- Test 2: 正常 batch（16）行为不变 ---
print("=== Test 2: normal batch (16) regression ===")
for cls, name in ((MosaicSurvFrozen, "AFiLM(frozen)"), (MosaicSurvTwostage, "BFiLM(twostage)")):
    model = cls(**make_kwargs("A" if name.startswith("A") else "B"))
    if name.startswith("A"):
        model.freeze_backbone_for_probe()
    else:
        model.freeze_backbone_keep_head()
    model.train()
    loss = run_batch(model, B=16)
    assert loss.requires_grad, f"{name}: 正常 batch loss 无梯度"
    loss.backward()
    print(f"OK: {name} normal-batch loss requires_grad=True")

# --- Test 3: variant C 小 batch 不崩（VAE 项保梯度，行为不变） ---
print("=== Test 3: variant C small batch still fine (unchanged) ===")
model_c = MosaicSurv(**make_kwargs("C"))
model_c.train()
surv_loss = run_batch(model_c, B=6)
combined = model_c.combine_loss(surv_loss, beta=0.0)
assert combined.requires_grad, "variant C 小 batch 联合损失无梯度"
combined.backward()
print("OK: variant C small-batch combined loss requires_grad=True")

print("\nALL TESTS PASSED")
