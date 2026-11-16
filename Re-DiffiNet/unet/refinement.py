"""KurtLabs encoder/denoiser with signed-residual x0 prediction and error gate.

Hard baseline b permits residual r=(1-2b)*m, m in [0,1]. The diffusion target
is y-b (NOT logits, and NOT 2*(y-b)-1). The exact same r is used by DDIM.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps


class DiffUNet(nn.Module):
    def __init__(self, width=32, sampling_steps=20):
        super().__init__()
        if width < 2 or not 2 <= sampling_steps <= 1000:
            raise ValueError('width >= 2 and sampling_steps in [2,1000] required')
        self.config = dict(width=width, sampling_steps=sampling_steps)
        features = [width, width, 2*width, 4*width, 8*width, width]
        self.embed_model = BasicUNetEncoder(3, 7, 3, features)
        self.model = BasicUNetDe(3, 10, 3, features,
                                act=('LeakyReLU', {'negative_slope': .1, 'inplace': False}))
        # Gate observes MRI/baseline features only, independently of noisy targets.
        self.gate_head = nn.Conv3d(width, 3, 1)
        nn.init.normal_(self.gate_head.weight, std=.01)
        nn.init.constant_(self.gate_head.bias, -2.)
        nn.init.normal_(self.model.final_conv.weight, std=.01)
        nn.init.constant_(self.model.final_conv.bias, -2.)
        args = dict(betas=get_named_beta_schedule('linear', 1000),
                    model_mean_type=ModelMeanType.START_X,
                    model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]), **args)
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [sampling_steps]), **args)

    @staticmethod
    def check_image(image):
        if image.ndim != 5 or image.shape[1] != 7:
            raise ValueError('Expected B,7,X,Y,Z')
        if any(s % 16 for s in image.shape[2:]) or min(image.shape[2:]) < 32:
            raise ValueError('Network patches must be multiples of 16 and >=32 per axis')

    def forward(self, x, t, image, embeddings=None):
        if embeddings is None:
            embeddings = self.embed_model(image)
        magnitude_logits = self.model(x, t, image=image, embeddings=embeddings)
        # FP32 decoding avoids rounding sigmoid outputs to 0/1 under BF16.
        return (1 - 2 * image[:, 4:].float()) * magnitude_logits.float().sigmoid()

    @staticmethod
    def compose(baseline, residual, gate):
        return (baseline.float() + gate.float() * residual.float()).clamp(0, 1)

    def training_losses(self, image, target, residual_weight=.5, gate_weight=.1, preservation_weight=.05):
        self.check_image(image)
        baseline = image[:, 4:].float()
        target = target.float()
        residual_target = target - baseline
        timesteps = torch.randint(0, 1000, (image.shape[0],), device=image.device)
        noisy = self.diffusion.q_sample(residual_target, timesteps, torch.randn_like(residual_target))
        embeddings = self.embed_model(image)
        residual = self(noisy, timesteps, image, embeddings)
        gate_logits = self.gate_head(embeddings[0]).float()
        gate = gate_logits.sigmoid()
        final = self.compose(baseline, residual, gate)
        # Explicit FP32 loss math, including BCE, even inside outer autocast.
        with torch.autocast(device_type=image.device.type, enabled=False):
            axes = tuple(range(2, final.ndim))
            dice = 1 - ((2*(final*target).sum(axes) + 1e-5) /
                        (final.sum(axes) + target.sum(axes) + 1e-5)).mean()
            bce = F.binary_cross_entropy(final.clamp(1e-6, 1-1e-6), target)
            residual_loss = F.mse_loss(residual, residual_target)
            gate_loss = F.binary_cross_entropy_with_logits(gate_logits, residual_target.abs())
            preservation = ((gate*residual).abs() * (1-residual_target.abs())).mean()
            total = dice + bce + residual_weight*residual_loss + gate_weight*gate_loss + preservation_weight*preservation
        return {'Loss': total, 'Dice_Loss': dice, 'BCE_Loss': bce,
                'Residual_Loss': residual_loss, 'Gate_BCE_Loss': gate_loss,
                'Preservation_Loss': preservation}

    @torch.no_grad()
    def sample(self, image, noise=None, return_parts=False):
        self.check_image(image)
        embeddings = self.embed_model(image)
        gate = self.gate_head(embeddings[0]).float().sigmoid()
        shape = (image.shape[0], 3, *image.shape[2:])
        if noise is None:
            noise = torch.randn(shape, device=image.device, dtype=torch.float32)
        if tuple(noise.shape) != shape:
            raise ValueError('Sampling noise shape mismatch')
        # Reuse original DDIM implementation, but retain only the last state.
        # Original ddim_sample_loop unnecessarily copies every 3D step to CPU.
        last = None
        for last in self.sample_diffusion.ddim_sample_loop_progressive(
                self, shape, noise=noise, device=image.device, eta=0.,
                model_kwargs={'image': image, 'embeddings': embeddings}):
            pass
        residual = last['pred_xstart']
        final = self.compose(image[:, 4:], residual, gate)
        return (final, residual, gate) if return_parts else final
