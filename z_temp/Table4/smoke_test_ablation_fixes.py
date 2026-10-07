"""Smoke test for the Table4 ablation fixes: a_film routing + surv0 grad blocking."""
import torch

from models.ablation_models.model_mosaic_surv_frozen import MosaicSurvFrozen
from models.ablation_models.model_mosaic_surv import MosaicSurv
from models.ablation_models.model_mosaic_surv_detached import MosaicSurvDetached

B = 16
N = 8


def fake_loss(h, t, c):
    return (h.view(-1) * t.view(-1)).mean()


def make_kwargs(variant):
    return dict(
        clinic_num_tokens=2,
        wsi_embedding_dim=1024,
        gene_embedding_dim=768,
        clinic_embedding_dim=512,
        gene_num_tokens=4,
        latent_dim=128,
        mmhid=256,
        label_dim=1,
        decoder_hidden_dim=512,
        poe_variant=variant,
        selected_modalities="wsi,gene,clinic",
    )


def inputs(avail_mask):
    x_path = torch.randn(B, N, 1024)
    x_omic = torch.randn(B, 4 * 768)
    x_clinic = torch.randn(B, 2 * 512)
    event_time = torch.rand(B, 1) * 100 + 1
    censor = torch.randint(0, 2, (B, 1)).float()
    avail = {
        "wsi": avail_mask[:, 0],
        "gene": avail_mask[:, 1],
        "clinic": avail_mask[:, 2],
    }
    return x_path, x_omic, x_clinic, event_time, censor, avail


def grad_none(module):
    return all(p.grad is None for p in module.parameters())


def grad_ok(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


# ---------- Test 1: AFiLM routes through FiLM head ----------
print("=== Test 1: AFiLM _survival_head routing ===")
model = MosaicSurvFrozen(**make_kwargs("A"))
model.eval()  # 关闭 Dropout，保证两次调用可比
mu = torch.randn(B, 128, requires_grad=True)
ones = torch.ones(B, 3, dtype=torch.bool)
risk, _ = model._survival_head(mu, ones)
risk_film = model.pattern_head(mu, pattern_id=7)
risk_probe = model.linear_probe(mu)
assert torch.allclose(risk, risk_film), "AFiLM forward did not route to FiLM head"
assert not torch.allclose(risk, risk_probe), "AFiLM routed to linear_probe (bug)"
print("OK: risk == FiLM head output, != linear_probe output")

# ---------- Test 2: AFiLM stage2 trains only the FiLM head ----------
print("=== Test 2: AFiLM stage2 gradient flow ===")
model.freeze_backbone_for_probe()
model.train()
x_path, x_omic, x_clinic, event_time, censor, avail = inputs(torch.ones(B, 3, dtype=torch.bool))
model(x_path=x_path, x_omic=x_omic, x_clinic=x_clinic, wsi_mask=None, avail=avail)
loss = model.survival_loss_from_cached(event_time, censor, fake_loss)
loss.backward()
assert grad_ok(model.pattern_head), "FiLM head got no gradient in AFiLM stage2"
assert grad_none(model.clinic_encoder), "encoder should be frozen in AFiLM stage2"
assert grad_none(model.linear_probe), "linear_probe should be frozen in AFiLM stage2"
assert grad_none(model.poe), "PoE alpha should be frozen in AFiLM stage2"
print("OK: only FiLM head receives gradient")

# ---------- Test 3: CFiLM (baseline) surv loss reaches encoder + alpha ----------
print("=== Test 3: CFiLM surv-loss gradient reaches backbone ===")
model_c = MosaicSurv(**make_kwargs("C"))
model_c.train()
model_c(x_path=x_path, x_omic=x_omic, x_clinic=x_clinic, wsi_mask=None, avail=avail)
loss = model_c.survival_loss_from_cached(event_time, censor, fake_loss)
loss.backward()
assert grad_ok(model_c.pattern_head), "FiLM head got no gradient"
assert grad_ok(model_c.clinic_encoder), "encoder should receive surv gradient in CFiLM"
assert grad_ok(model_c.poe), "PoE alpha should receive surv gradient in CFiLM"
print("OK: head + encoder + alpha all receive gradient")

# ---------- Test 4: surv0 blocks surv gradient from encoder + alpha ----------
print("=== Test 4: CFiLMNoSurvGrad blocks surv gradient from backbone ===")
model_s = MosaicSurvDetached(**make_kwargs("C"))
model_s.train()
model_s(x_path=x_path, x_omic=x_omic, x_clinic=x_clinic, wsi_mask=None, avail=avail)
loss = model_s.survival_loss_from_cached(event_time, censor, fake_loss)
loss.backward()
assert grad_ok(model_s.pattern_head), "FiLM head should still receive surv gradient in surv0"
assert grad_none(model_s.clinic_encoder), "encoder should NOT receive surv gradient in surv0"
assert grad_none(model_s.poe), "PoE alpha should NOT receive surv gradient in surv0"
print("OK: head receives gradient, encoder/alpha blocked")

# ---------- Test 5: surv0 VAE loss still reaches backbone ----------
print("=== Test 5: surv0 VAE loss still trains backbone ===")
model_s.zero_grad()
vae_loss = model_s.get_vae_loss(beta=1.0)
vae_loss.backward()
assert grad_ok(model_s.clinic_encoder), "encoder should receive VAE gradient in surv0"
assert grad_ok(model_s.poe), "PoE alpha should receive VAE gradient in surv0"
print("OK: L_rec + beta*J still reach encoder and alpha")

print("\nALL SMOKE TESTS PASSED")
