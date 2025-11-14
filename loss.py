import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.blocks import Convolution

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class Medical3DSILoss:
    def __init__(
        self,
        prediction='epsilon',
        path_type="linear",
        weighting="uniform",
        encoders=[], 
        encoder_types=[],
        device=None,
        proj_coefficient=0.5,
        inpainting_mode=True,
        mask_region_weight=1.0,
        unmask_region_weight=0.1,
    ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.encoder_types = encoder_types
        self.device = device
        self.proj_coefficient = proj_coefficient
        self.inpainting_mode = inpainting_mode
        self.mask_region_weight = mask_region_weight
        self.unmask_region_weight = unmask_region_weight
        
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def extract_encoder_features(self, images, mask=None):
        features_list = []
        
        with torch.no_grad():
            for encoder, encoder_type in zip(self.encoders, self.encoder_types):
                if '3d' in encoder_type.lower():
                    features = encoder(images)
                else:
                    batch_size, channels, depth, height, width = images.shape
                    slice_features = []
                    
                    for d in range(0, depth, max(1, depth // 8)):
                        slice_img = images[:, :, d, :, :]
                        if slice_img.shape[1] == 1:
                            slice_img = slice_img.repeat(1, 3, 1, 1)
                        slice_feat = encoder(slice_img)
                        slice_features.append(slice_feat)
                    
                    features = torch.stack(slice_features, dim=2)
                    features = features.mean(dim=2)
                
                features_list.append(features)
        
        return features_list
    
    def compute_projection_loss(self, z_pred_list, z_target_list, mask=None):
        proj_loss = 0.
        total_pairs = 0
        
        for z_pred, z_target in zip(z_pred_list, z_target_list):
            if z_pred.shape != z_target.shape:
                if len(z_pred.shape) > len(z_target.shape):
                    z_target = z_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    z_target = z_target.expand_as(z_pred)
                elif len(z_target.shape) > len(z_pred.shape):
                    z_target = F.adaptive_avg_pool3d(z_target, (1, 1, 1)).flatten(2)
                    z_pred = F.adaptive_avg_pool3d(z_pred, (1, 1, 1)).flatten(2)
            
            z_pred_norm = F.normalize(z_pred, dim=1)
            z_target_norm = F.normalize(z_target, dim=1)
            
            cosine_sim = (z_pred_norm * z_target_norm).sum(dim=1)
            
            if self.inpainting_mode and mask is not None:
                if len(cosine_sim.shape) > 1:
                    mask_resized = F.interpolate(
                        mask.float(), 
                        size=cosine_sim.shape[1:], 
                        mode='trilinear', 
                        align_corners=False
                    )
                    
                    mask_loss = mean_flat(-cosine_sim * mask_resized.squeeze(1))
                    unmask_loss = mean_flat(-cosine_sim * (1 - mask_resized.squeeze(1)))
                    
                    weighted_loss = (self.mask_region_weight * mask_loss + 
                                     self.unmask_region_weight * unmask_loss)
                else:
                    weighted_loss = mean_flat(-cosine_sim)
            else:
                weighted_loss = mean_flat(-cosine_sim)
            
            proj_loss += weighted_loss
            total_pairs += 1
        
        if total_pairs > 0:
            proj_loss = proj_loss / total_pairs
        
        return proj_loss
    
    def __call__(self, model, images, model_kwargs=None, target_features=None, mask=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1, 1))
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        timesteps = time_input.flatten()
        
        noise = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        
        if self.inpainting_mode and mask is not None:
            noisy_images = images.clone()
            noisy_mask_region = alpha_t * images + sigma_t * noise
            noisy_images = (1 - mask) * images + mask * noisy_mask_region
        else:
            noisy_images = alpha_t * images + sigma_t * noise
        
        model_kwargs_with_mask = model_kwargs.copy()
        if self.inpainting_mode and mask is not None:
            model_kwargs_with_mask['inpaint_mask'] = mask
        
        model_output, pred_features = model(
            noisy_images, 
            timesteps, 
            **model_kwargs_with_mask,
            return_features=True
        )
        
        if self.prediction == 'epsilon':
            target = noise
        elif self.prediction == 'v':
            target = d_alpha_t * images + d_sigma_t * noise
        elif self.prediction == 'sample':
            target = images
        else:
            raise NotImplementedError(f"Unknown prediction type: {self.prediction}")
        
        if self.inpainting_mode and mask is not None:
            mask_loss = mean_flat((model_output * mask - target * mask) ** 2)
            unmask_loss = mean_flat((model_output * (1 - mask) - target * (1 - mask)) ** 2)
            denoising_loss = mask_loss + 0.1 * unmask_loss
        else:
            denoising_loss = mean_flat((model_output - target) ** 2)
        
        if target_features is not None and pred_features is not None:
            proj_loss = self.compute_projection_loss(pred_features, target_features, mask)
        else:
            proj_loss = torch.tensor(0.0, device=images.device)
        
        return denoising_loss, proj_loss


class MedicalImageProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                norm="batch",
                act="silu"
            ),
            Convolution(
                spatial_dims=3,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                norm="batch",
                act="silu"
            ),
            Convolution(
                spatial_dims=3,
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=1,
                conv_only=True
            ),
        )
    
    def forward(self, x):
        return self.projector(x)