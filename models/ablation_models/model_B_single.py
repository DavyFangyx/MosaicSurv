"""Model B + single-head: freeze enc/dec, refill missing modalities, then recode."""

from models.model_SurvTriPoEVAE import SurvTriPoEVAE
from models.model_utils import ModalityDecoder, ReconstructionLoss, refill_missing_modalities, reparameterize
import torch


class SurvTriPoEVAE_BSingle(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        kwargs["poe_variant"] = "B"
        super().__init__(*args, **kwargs)
        wsi_dim = int(self.wsi_resampler_tokens) * 768
        self.decoder_wsi = ModalityDecoder(self.latent_dim, self.decoder_hidden_dim, wsi_dim)
        self.reconstruction_loss = ReconstructionLoss({
            "wsi": wsi_dim,
            "gene": self.gene_num_tokens * self.gene_embedding_dim,
            "clinic": self.clinic_num_tokens * self.clinic_embedding_dim,
        })

    def _wsi_reconstruction_target(self, wsi_tokens):
        return wsi_tokens.reshape(wsi_tokens.shape[0], -1)

    def forward(self, x_path, x_omic, x_clinic, wsi_mask=None, avail=None):
        if not (self.training_stage == "stage2" and self.backbone_frozen):
            return super().forward(x_path, x_omic, x_clinic, wsi_mask=wsi_mask, avail=avail)

        x_omic = self._reshape_gene(x_omic.float())
        x_clinic = self._reshape_clinic(x_clinic.float())
        x_path = x_path.float()
        available_mask = self._normalize_avail(avail, x_path.device)
        batch_size = x_path.shape[0]

        wsi_tokens = self.wsi_resampler(x_path, padding_mask=wsi_mask)
        mu_list, logvar_list = self._encode_tokens(wsi_tokens, x_omic, x_clinic)
        mu_joint, logvar_joint, poe_weights = self.poe(
            mus=mu_list,
            logvars=logvar_list,
            available_mask=available_mask,
        )
        z_joint = reparameterize(mu_joint, logvar_joint, sample=False)
        recon_wsi, recon_gene, recon_clinic = self._decode(z_joint)

        target_wsi = self._wsi_reconstruction_target(wsi_tokens)
        recon_losses = self.reconstruction_loss(
            recon_dict={"wsi": recon_wsi, "gene": recon_gene, "clinic": recon_clinic},
            target_dict={
                "wsi": target_wsi,
                "gene": x_omic.reshape(batch_size, -1),
                "clinic": x_clinic.reshape(batch_size, -1),
            },
        )
        jeffreys = self.jeffreys(mu_joint, logvar_joint)

        complete = available_mask.all(dim=1)
        if (~complete).any():
            with torch.no_grad():
                wsi_tokens_2, gene_tokens_2, clinic_tokens_2 = refill_missing_modalities(
                    wsi_tokens, x_omic, x_clinic, recon_wsi, recon_gene, recon_clinic, available_mask,
                )
                mu_list_2, logvar_list_2 = self._encode_tokens(wsi_tokens_2, gene_tokens_2, clinic_tokens_2)
                full_mask = torch.ones_like(available_mask)
                mu_joint_2, logvar_joint_2, _ = self.poe(
                    mus=mu_list_2,
                    logvars=logvar_list_2,
                    available_mask=full_mask,
                )
            mu_joint = torch.where(complete.unsqueeze(1), mu_joint, mu_joint_2)
            logvar_joint = torch.where(complete.unsqueeze(1), logvar_joint, logvar_joint_2)

        z_joint = mu_joint
        risk, fused = self._survival_head(mu_joint, available_mask)
        self._cached_outputs = {
            "risk": risk,
            "fused": fused,
            "z_joint": z_joint,
            "mu_joint": mu_joint,
            "logvar_joint": logvar_joint,
            "mu_list": mu_list,
            "logvar_list": logvar_list,
            "poe_weights": poe_weights,
            "available_mask": available_mask,
            "wsi_tokens": wsi_tokens,
            "recon_losses": recon_losses,
            "recon_total": recon_losses["total"],
            "jeffreys": jeffreys,
        }
        return risk

