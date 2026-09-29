import torch
import torch.nn as nn

from models.model_utils import (
    GeneralizedPoE,
    KLDivergence,
    JeffreysDivergence,
    ModalityDecoder,
    ReconstructionLoss,
    TokenSetEncoder,
    WSIMILResampler,
    WSITargetPoolingHead,
    bits_to_id,
    multi_pattern_surv_step,
    parse_alphapgc,
    reparameterize,
)

class SurvTriPoEVAE(nn.Module):
    def __init__(
        self,
        clinic_num_tokens,
        wsi_embedding_dim=1024,
        gene_embedding_dim=768,
        clinic_embedding_dim=512,
        gene_num_tokens=4,
        latent_dim=128,
        mmhid=256,
        label_dim=1,
        decoder_hidden_dim=512,
        poe_variant="A",
        poe_surv_lambda=1.0,
        modality_dropout_prob=0.2,
        transformer_dropout=0.1,
        transformer_layers=1,
        wsi_resampler_tokens=16,
        wsi_resampler_layers=2,
        selected_modalities="wsi,gene,clinic",
        alphafix=False,
        alphapgc=None,
    ):
        super().__init__()
        self.gene_num_tokens = gene_num_tokens
        self.clinic_num_tokens = clinic_num_tokens
        self.wsi_embedding_dim = wsi_embedding_dim
        self.gene_embedding_dim = gene_embedding_dim
        self.clinic_embedding_dim = clinic_embedding_dim
        self.latent_dim = latent_dim
        self.mmhid = mmhid
        self.label_dim = label_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.poe_variant = poe_variant.upper()
        self.poe_surv_lambda = poe_surv_lambda
        self.modality_dropout_prob = modality_dropout_prob
        self.modality_names = ("wsi", "gene", "clinic")
        self.selected_modalities = tuple(selected_modalities.split(","))

        if self.poe_variant not in {"A", "B", "C"}:
            raise ValueError(f"Unsupported poe_variant `{poe_variant}`.")
        if not self.selected_modalities:
            raise ValueError("At least one modality must be selected.")
        if any(name not in self.modality_names for name in self.selected_modalities):
            raise ValueError(f"Unsupported selected_modalities `{selected_modalities}`.")

        self.training_stage = "stage1" if self.poe_variant in {"A", "B"} else "stage2"
        self.backbone_frozen = False
        self.head_trainable = False
        self.pattern_head = None
        self.uses_multi_pattern_surv = False
        self.surv_detach_poe = False
        self.wsi_resampler_tokens = wsi_resampler_tokens

        self.wsi_resampler = WSIMILResampler(
            input_dim=wsi_embedding_dim,
            token_dim=768,
            num_tokens=wsi_resampler_tokens,
            nhead=8,
            mlp_dim=1024,
            num_layers=wsi_resampler_layers,
            dropout=transformer_dropout,
        )
        self.wsi_encoder = TokenSetEncoder(
            input_dim=768,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )
        self.gene_encoder = TokenSetEncoder(
            input_dim=gene_embedding_dim,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )
        self.clinic_encoder = TokenSetEncoder(
            input_dim=clinic_embedding_dim,
            latent_dim=latent_dim,
            nhead=8,
            mlp_dim=1024,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )

        self.wsi_target_head = WSITargetPoolingHead(token_dim=768, output_dim=768)
        self.alphafix = bool(alphafix)
        self.alphapgc = ",".join(f"{weight:g}" for weight in parse_alphapgc(alphapgc)) if self.alphafix else alphapgc
        self.poe = GeneralizedPoE(num_modalities=3, alphafix=self.alphafix, alphapgc=alphapgc)

        self.decoder_wsi = ModalityDecoder(latent_dim, decoder_hidden_dim, 768)
        self.decoder_gene = ModalityDecoder(latent_dim, decoder_hidden_dim, gene_num_tokens * gene_embedding_dim)
        self.decoder_clinic = ModalityDecoder(latent_dim, decoder_hidden_dim, clinic_num_tokens * clinic_embedding_dim)

        self.reconstruction_loss = ReconstructionLoss({
            "wsi": 768,
            "gene": gene_num_tokens * gene_embedding_dim,
            "clinic": clinic_num_tokens * clinic_embedding_dim,
        })
        self.jeffreys = JeffreysDivergence()

        self.fuse_fc = nn.Sequential(
            nn.Linear(latent_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(mmhid, label_dim)
        self.linear_probe = nn.Linear(latent_dim, label_dim)

        self._cached_outputs = None

    def _disable_legacy_survival_head(self):
        for module in (self.fuse_fc, self.classifier, self.linear_probe):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            for module in self._backbone_modules():
                module.eval()
            if self.head_trainable:
                if self.pattern_head is not None:
                    self.pattern_head.train(mode)
                    self.fuse_fc.eval()
                    self.classifier.eval()
                else:
                    self.fuse_fc.train(mode)
                    self.classifier.train(mode)
            else:
                self.fuse_fc.eval()
                self.classifier.eval()
                if self.pattern_head is not None:
                    self.pattern_head.eval()
            self.linear_probe.train(mode)
        return self

    def set_training_stage(self, stage):
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported training stage `{stage}`.")
        self.training_stage = stage

    def get_stage2_param_groups(self, head_lr, encoder_lr_ratio):
        encoder_modules = [
            self.wsi_resampler,
            self.wsi_encoder,
            self.gene_encoder,
            self.clinic_encoder,
            self.wsi_target_head,
            self.poe,
            self.decoder_wsi,
            self.decoder_gene,
            self.decoder_clinic,
            self.reconstruction_loss,
        ]
        head_modules = [self.fuse_fc, self.classifier]

        def _trainable(modules):
            params = []
            for module in modules:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        params.append(parameter)
            return params

        encoder_params = _trainable(encoder_modules)
        head_params = _trainable(head_modules)
        assigned = {id(parameter) for parameter in encoder_params + head_params}
        leftover = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in assigned and not name.startswith("linear_probe.")
        ]
        if leftover:
            raise ValueError(
                "Unassigned trainable parameters in Model B stage2: " + ", ".join(leftover)
            )
        encoder_lr = float(head_lr) * float(encoder_lr_ratio)
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            groups.append({"params": head_params, "lr": float(head_lr)})
        if not groups:
            raise ValueError("No trainable parameters for Model B stage2.")
        return groups

    def freeze_backbone_for_probe(self):
        modules = [
            self.wsi_resampler,
            self.wsi_encoder,
            self.gene_encoder,
            self.clinic_encoder,
            self.wsi_target_head,
            self.poe,
            self.decoder_wsi,
            self.decoder_gene,
            self.decoder_clinic,
            self.reconstruction_loss,
        ]
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        for parameter in self.linear_probe.parameters():
            parameter.requires_grad = True

        for parameter in self.fuse_fc.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

        self.backbone_frozen = True
        self.head_trainable = False
        self.training_stage = "stage2"

    def _backbone_modules(self):
        modules = [
            self.wsi_resampler,
            self.wsi_encoder,
            self.gene_encoder,
            self.clinic_encoder,
            self.poe,
            self.decoder_wsi,
            self.decoder_gene,
            self.decoder_clinic,
            self.reconstruction_loss,
            self.linear_probe,
        ]
        if getattr(self, "wsi_target_head", None) is not None:
            modules.append(self.wsi_target_head)
        return modules

    def freeze_backbone_keep_head(self):
        for module in self._backbone_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        head_modules = [self.pattern_head] if self.pattern_head is not None else [self.fuse_fc, self.classifier]
        for module in head_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if self.pattern_head is not None:
            for module in (self.fuse_fc, self.classifier, self.linear_probe):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        self.backbone_frozen = True
        self.head_trainable = True
        self.training_stage = "stage2"

    def get_cached_outputs(self):
        if self._cached_outputs is None:
            raise RuntimeError("No cached outputs available. Run forward first.")
        return self._cached_outputs

    def get_vae_loss(self, beta=1.0):
        cached = self.get_cached_outputs()
        return cached["recon_total"] + beta * cached["jeffreys"]

    def combine_loss(self, survival_loss, beta=1.0):
        if self.poe_variant == "C":
            return self.get_vae_loss(beta=beta) + self.poe_surv_lambda * survival_loss
        return survival_loss

    def survival_loss_from_cached(self, event_time, censor, loss_fn, avail_real=None, min_batch_for_cox=8):
        cached = self.get_cached_outputs()
        if not self.uses_multi_pattern_surv:
            return loss_fn(h=cached["risk"], t=event_time, c=censor)
        if avail_real is None:
            avail_real = torch.ones_like(cached["available_mask"])
        result = multi_pattern_surv_step(
            mu_list=cached["mu_list"],
            logvar_list=cached["logvar_list"],
            avail_real=avail_real,
            poe=self.poe,
            head=self.pattern_head,
            event_time=event_time,
            censor=censor,
            min_batch_for_cox=min_batch_for_cox,
            loss_fn=loss_fn,
            stop_grad_poe=self.surv_detach_poe,
        )
        cached["pattern_surv"] = result
        mu_joint_pgc, _, _ = self.poe(
            mus=cached["mu_list"],
            logvars=cached["logvar_list"],
            available_mask=torch.ones_like(cached["available_mask"]),
        )
        cached["risk"] = self.pattern_head(mu_joint_pgc, pattern_id=7)
        return result["loss"]

    def _reshape_gene(self, x_omic):
        if x_omic.dim() == 2:
            return x_omic.reshape(x_omic.shape[0], self.gene_num_tokens, self.gene_embedding_dim)
        return x_omic

    def _reshape_clinic(self, x_clinic):
        if x_clinic.dim() == 2:
            return x_clinic.reshape(x_clinic.shape[0], self.clinic_num_tokens, self.clinic_embedding_dim)
        return x_clinic

    def _normalize_avail(self, avail, device):
        if avail is None:
            raise ValueError("survtri_poe_vae now requires dataloader-provided `avail`.")
        normalized = []
        for name in self.modality_names:
            if name not in avail:
                raise KeyError(f"Missing modality availability key `{name}`.")
            value = avail[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.bool, device=device)
            else:
                value = value.to(device=device, dtype=torch.bool)
            normalized.append(value.view(-1))
        return torch.stack(normalized, dim=1)

    def _select_latent_for_survival(self, mu_joint, z_joint):
        if self.training_stage == "stage1":
            return z_joint
        if self.poe_variant == "A":
            return mu_joint
        if self.training:
            return z_joint
        return mu_joint

    def _wsi_reconstruction_target(self, wsi_tokens):
        return self.wsi_target_head(wsi_tokens)

    def _encode_tokens(self, wsi_tokens, gene_tokens, clinic_tokens):
        mu_wsi, logvar_wsi = self.wsi_encoder(wsi_tokens)
        mu_gene, logvar_gene = self.gene_encoder(gene_tokens)
        mu_clinic, logvar_clinic = self.clinic_encoder(clinic_tokens)
        mu_list = [mu_wsi, mu_gene, mu_clinic]
        logvar_list = [logvar_wsi, logvar_gene, logvar_clinic]
        return mu_list, logvar_list

    def _decode(self, z_joint):
        recon_wsi = self.decoder_wsi(z_joint)
        recon_gene = self.decoder_gene(z_joint)
        recon_clinic = self.decoder_clinic(z_joint)
        return recon_wsi, recon_gene, recon_clinic

    def _survival_head(self, survival_latent, available_mask):
        # pattern_head 优先于 variant A 的 linear_probe，保证 AFiLM 走 FiLM 头
        if self.pattern_head is not None:
            pattern_id = bits_to_id(available_mask)
            risk = self.pattern_head(survival_latent, pattern_id)
            return risk, survival_latent
        if self.poe_variant == "A":
            fused = survival_latent
            risk = self.linear_probe(survival_latent)
            return risk, fused
        fused = self.fuse_fc(survival_latent)
        risk = self.classifier(fused)
        return risk, fused

    def forward(self, x_path, x_omic, x_clinic, wsi_mask=None, avail=None):
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

        should_sample = self.training and self.poe_variant in {"B", "C"}
        if self.training_stage == "stage1":
            should_sample = True
        z_joint = reparameterize(mu_joint, logvar_joint, sample=should_sample)

        recon_wsi, recon_gene, recon_clinic = self._decode(z_joint)

        target_wsi = self._wsi_reconstruction_target(wsi_tokens)
        target_gene = x_omic.reshape(batch_size, -1)
        target_clinic = x_clinic.reshape(batch_size, -1)

        recon_losses = self.reconstruction_loss(
            recon_dict={
                "wsi": recon_wsi,
                "gene": recon_gene,
                "clinic": recon_clinic,
            },
            target_dict={
                "wsi": target_wsi,
                "gene": target_gene,
                "clinic": target_clinic,
            },
        )
        jeffreys = self.jeffreys(mu_joint, logvar_joint)
        survival_latent = self._select_latent_for_survival(mu_joint, z_joint)
        risk, fused = self._survival_head(survival_latent, available_mask)

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


class SurvTriPoEVAE_KL(SurvTriPoEVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jeffreys = KLDivergence()
