from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from monai.apps.generation.maisi.networks import diffusion_model_unet_maisi
from torch.nn.parallel import DistributedDataParallel
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance
import matplotlib.pyplot as plt
import numpy as np
from monai.visualize import plot_2d_or_3d_image
import torchvision.utils as vutils
import os.path as osp
from monai.data import NibabelWriter
from monai.transforms import SqueezeDim
import nibabel as nib
from .new_utils import calculate_scale_factor,load_unet,load_filenames,create_lr_scheduler,create_optimizer
from .PASTA.classification.networks.generic_UNet import Generic_UNet_classify
import torch.nn as nn
from .PASTA.classification.dataloaders.data_process_func import img_multi_thresh_normalized_torch
import torch.nn.functional as F

def get_pasta():
    num_input_channels = 1
    base_num_features = 64
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_num_pool_op_kernel_sizes = net_conv_kernel_sizes = None

    net = Generic_UNet_classify(num_input_channels, base_num_features, 1, 5,
                                 conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                 dropout_op_kwargs,
                                 net_nonlin, net_nonlin_kwargs, True, False,
                                 net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float().cuda()
    return net

def load_pasta_model(checkpoint_path, device):
    logger = logging.getLogger("training")
    logger.info(f"正在从 {checkpoint_path} 加载PASTA模型...")
    
    pasta_model = get_pasta()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PASTA checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_dict = pasta_model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(filtered_dict)
    pasta_model.load_state_dict(model_dict, strict=False)
    
    skipped_params = set(checkpoint.keys()) - set(filtered_dict.keys())
    if skipped_params:
        logger.warning(f"PASTA加载时跳过了 {len(skipped_params)} 个参数 (通常是分类头，这对于特征提取是正常的): {skipped_params}")

    pasta_model = pasta_model.to(device)
    pasta_model.eval()
    
    for param in pasta_model.parameters():
        param.requires_grad = False
        
    logger.info("PASTA模型加载并冻结成功。")
    return pasta_model

def pasta_preprocess_fn(img_tensor):
    thresh_ls = [-1000, -200, 200, 1000]
    norm_ls = [0, 0.2, 0.8, 1]
    return img_multi_thresh_normalized_torch(img_tensor, thresh_ls, norm_ls, data_type=torch.float32)

class DualImageAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, plain_features, enhanced_features):
        combined = torch.cat([plain_features, enhanced_features], dim=-1)
        weights = self.attention(combined)
        
        fused = (weights[:, 0:1] * plain_features + 
                 weights[:, 1:2] * enhanced_features)
        return fused

def extract_dual_pasta_features(
    pasta_model, 
    plain_images, 
    enhanced_images, 
    bbox_masks,
    pasta_target_size=(64, 64, 64),
    pasta_padding=5,
    fusion_method="concat",
    enhanced_weight=0.7,
    attention_fusion_layer=None
):
    batch_size = plain_images.shape[0]
    dual_features_list = []
    
    with torch.no_grad():
        for b in range(batch_size):
            plain_cropped = crop_image_by_bbox_mask(
                plain_images[b], bbox_masks[b], 
                target_size=pasta_target_size, 
                padding=pasta_padding
            )
            enhanced_cropped = crop_image_by_bbox_mask(
                enhanced_images[b], bbox_masks[b],
                target_size=pasta_target_size, 
                padding=pasta_padding
            )
            
            plain_processed = pasta_preprocess_fn(plain_cropped.unsqueeze(0))
            enhanced_processed = pasta_preprocess_fn(enhanced_cropped.unsqueeze(0))
            
            plain_features = pasta_model(plain_processed, output_feature=True)
            enhanced_features = pasta_model(enhanced_processed, output_feature=True)
            
            if fusion_method == "concat":
                fused_features = torch.cat([plain_features, enhanced_features], dim=-1)
                
            elif fusion_method == "add":
                fused_features = plain_features + enhanced_features
                
            elif fusion_method == "weighted":
                fused_features = (enhanced_weight * enhanced_features + 
                                  (1 - enhanced_weight) * plain_features)
                
            elif fusion_method == "attention":
                if attention_fusion_layer is None:
                    raise ValueError("attention_fusion_layer is required for attention fusion")
                fused_features = attention_fusion_layer(plain_features, enhanced_features)
                
            else:
                raise ValueError(f"Unsupported fusion method: {fusion_method}")
            
            dual_features_list.append(fused_features)
    
    return torch.cat(dual_features_list, dim=0)

def crop_image_by_bbox_mask(image, bbox_mask, target_size=(64, 64, 64), padding=5):
    mask_single = bbox_mask[0]
    nonzero_indices = torch.nonzero(mask_single, as_tuple=False)
    
    if len(nonzero_indices) == 0:
        d, h, w = mask_single.shape
        z_min, z_max = d//4, 3*d//4
        y_min, y_max = h//4, 3*h//4
        x_min, x_max = w//4, 3*w//4
    else:
        z_min, y_min, x_min = nonzero_indices.min(dim=0)[0]
        z_max, y_max, x_max = nonzero_indices.max(dim=0)[0]
        
        d, h, w = mask_single.shape
        z_min = max(0, z_min - padding)
        z_max = min(d, z_max + padding + 1)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding + 1)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding + 1)
    
    cropped = image[:, z_min:z_max, y_min:y_max, x_min:x_max]
    
    cropped_resized = F.interpolate(
        cropped.unsqueeze(0),
        size=target_size, 
        mode='trilinear', 
        align_corners=True
    ).squeeze(0)
    
    return cropped_resized

def prepare_data_from_json_dual_pasta(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    args: argparse.Namespace,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
    inpainting_mode: bool = False,
    use_dual_pasta: bool = True,
) -> DataLoader:
    data_dicts = []
    data_root = args.embedding_base_dir if hasattr(args, 'embedding_base_dir') else args.data_root
    
    print(f"数据准备开始: inpainting_mode={inpainting_mode}, use_dual_pasta={use_dual_pasta}")
    print(f"数据根目录: {data_root}")
    print(f"总共有 {len(train_files)} 个训练文件")
    
    skipped_count = 0
    processed_count = 0
    
    for item in train_files:
        plain_image_path = item.get("plain_image")
        if not plain_image_path:
            print(f"Warning: No plain_image found in item: {item}")
            skipped_count += 1
            continue
            
        if os.path.isabs(plain_image_path):
            base_path = os.path.dirname(plain_image_path)
        else:
            image_dirname = os.path.dirname(plain_image_path)
            base_path = os.path.join(data_root, image_dirname) if image_dirname else data_root
            
        latent_path = plain_image_path
        if not os.path.isabs(latent_path):
            latent_path = os.path.join(data_root, latent_path)
            
        enhanced_image_path = None
        if use_dual_pasta and "enhanced_image" in item:
            enhanced_image_path = item["enhanced_image"]
            if not os.path.isabs(enhanced_image_path):
                enhanced_image_path = os.path.join(data_root, enhanced_image_path)
            if not os.path.exists(enhanced_image_path):
                print(f"Warning: Enhanced image not found at {enhanced_image_path}")
                enhanced_image_path = None
                
        enhanced_label_path = None
        if use_dual_pasta and "enhanced_label" in item:
            enhanced_label_path = item["enhanced_label"]
            if not os.path.isabs(enhanced_label_path):
                enhanced_label_path = os.path.join(data_root, enhanced_label_path)
            if not os.path.exists(enhanced_label_path):
                print(f"Warning: Enhanced label not found at {enhanced_label_path}")
                enhanced_label_path = None
                
        plain_label_path = item.get("plain_label", None)
        if plain_label_path and not os.path.isabs(plain_label_path):
            plain_label_path = os.path.join(data_root, plain_label_path)
        
        if not os.path.exists(latent_path):
            print(f"无法找到潜在嵌入文件: {plain_image_path}")
            skipped_count += 1
            continue
        
        if inpainting_mode and (plain_label_path is None or not os.path.exists(plain_label_path)):
            print(f"Inpainting模式需要plain_label文件, 但{item.get('plain_label', '未指定')}不存在")
            skipped_count += 1
            continue
            
        if use_dual_pasta:
            plain_raw_path = latent_path.replace("_emb.nii.gz", ".nii.gz")
            if not os.path.exists(plain_raw_path):
                print(f"双输入PASTA模式需要plain原始图像，但{plain_raw_path}不存在")
                skipped_count += 1
                continue
                
            if enhanced_image_path is None or enhanced_label_path is None:
                print(f"双输入PASTA模式需要enhanced_image和enhanced_label文件，但有文件缺失")
                print(f"  enhanced_image: {enhanced_image_path if enhanced_image_path else 'MISSING'}")
                print(f"  enhanced_label: {enhanced_label_path if enhanced_label_path else 'MISSING'}")
                skipped_count += 1
                continue
            
        data_dict = {
            "image": latent_path,
            "meta_dict": item,
            "class_label": torch.tensor(item["class_label"], dtype=torch.long),
            "spacing": item.get("spacing", None) 
        }
        
        if use_dual_pasta:
            data_dict["plain_raw_image"] = plain_raw_path
            data_dict["enhanced_image"] = enhanced_image_path
            data_dict["enhanced_label"] = enhanced_label_path
        
        if plain_label_path and os.path.exists(plain_label_path):
            data_dict["plain_label"] = plain_label_path
        
        data_dicts.append(data_dict)
        processed_count += 1
    
    print(f"数据准备完成: 处理了 {processed_count} 个文件，跳过了 {skipped_count} 个文件")
    
    if len(data_dicts) == 0:
        raise RuntimeError("没有有效的训练数据！请检查数据路径和文件是否存在。")
    
    class CreateBboxMaskFromEnhancedLabeld(monai.transforms.MapTransform):
        def __init__(self, keys, enhanced_label_key="enhanced_label", bbox_mask_key="bbox_mask", bbox_value=2):
            super().__init__(keys)
            self.enhanced_label_key = enhanced_label_key
            self.bbox_mask_key = bbox_mask_key
            self.bbox_value = bbox_value
        
        def __call__(self, data):
            d = dict(data)
            if self.enhanced_label_key not in d:
                enhanced_image = d.get("enhanced_image")
                if enhanced_image is not None:
                    bbox_mask = torch.zeros_like(enhanced_image[[0]], dtype=torch.float32)
                    d[self.bbox_mask_key] = bbox_mask
                return d
            
            enhanced_label = d[self.enhanced_label_key]
            if hasattr(enhanced_label, 'data'):
                enhanced_label = enhanced_label.data
            elif isinstance(enhanced_label, monai.data.MetaTensor):
                enhanced_label = enhanced_label.as_tensor()
                
            bbox_mask = (enhanced_label == self.bbox_value).float()
            if isinstance(bbox_mask, monai.data.MetaTensor):
                bbox_mask = bbox_mask.as_tensor()
            
            d[self.bbox_mask_key] = bbox_mask
            return d
    
    class CreateInpaintMaskFromPlainLabeld(monai.transforms.MapTransform):
        def __init__(self, keys, plain_label_key="plain_label", inpaint_mask_key="inpaint_mask", mask_value=2):
            super().__init__(keys)
            self.plain_label_key = plain_label_key
            self.inpaint_mask_key = inpaint_mask_key
            self.mask_value = mask_value
        
        def __call__(self, data):
            d = dict(data)
            if self.plain_label_key not in d:
                raise KeyError(f"Plain label key {self.plain_label_key} not found in data")
            
            plain_label = d[self.plain_label_key]
            if hasattr(plain_label, 'data'):
                plain_label = plain_label.data
            elif isinstance(plain_label, monai.data.MetaTensor):
                plain_label = plain_label.as_tensor()
                
            inpaint_mask = (plain_label == self.mask_value).float()
            if isinstance(inpaint_mask, monai.data.MetaTensor):
                inpaint_mask = inpaint_mask.as_tensor()
                
            d[self.inpaint_mask_key] = inpaint_mask
            return d
    
    class ExtractMetaDatad(monai.transforms.MapTransform):
        def __init__(self, keys, extract_key):
            super().__init__(keys)
            self.extract_key = extract_key
        
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                meta_dict = d["meta_dict"]
                if self.extract_key == "spacing":
                    d[key] = torch.FloatTensor(meta_dict["spacing"]) * 1e2
                elif self.extract_key == "top_region_index":
                    d[key] = torch.FloatTensor(meta_dict["top_region_index"]) * 1e2
                elif self.extract_key == "bottom_region_index":
                    d[key] = torch.FloatTensor(meta_dict["bottom_region_index"]) * 1e2
            return d
    
    keys_to_load = ["image"]
    dual_pasta_keys = []
    
    if use_dual_pasta:
        dual_pasta_keys = ["plain_raw_image", "enhanced_image", "enhanced_label"]
        print(f"Dual PASTA enabled: loading {dual_pasta_keys}")
        
    if inpainting_mode:
        keys_to_load.append("plain_label")
        print(f"Inpainting mode enabled: loading {keys_to_load}")
        
    print(f"Final keys to load: {keys_to_load + dual_pasta_keys}")
        
    train_transforms_list = [
        monai.transforms.LoadImaged(keys=keys_to_load + dual_pasta_keys),
        monai.transforms.EnsureChannelFirstd(keys=keys_to_load + dual_pasta_keys),
        ExtractMetaDatad(keys=["spacing"], extract_key="spacing"),
    ]
    
    if include_body_region:
        train_transforms_list += [
            ExtractMetaDatad(keys=["top_region_index"], extract_key="top_region_index"),
            ExtractMetaDatad(keys=["bottom_region_index"], extract_key="bottom_region_index"),
        ]
    
    target_size = [32, 32, 32]
    train_transforms_list.append(
        monai.transforms.Resized(
            keys=["image"] + (["plain_label"] if inpainting_mode else []),
            spatial_size=target_size,
            mode=("trilinear", "nearest") if inpainting_mode else "trilinear",
            align_corners=(True, None) if inpainting_mode else True,
        )
    )
    
    if use_dual_pasta:
        train_transforms_list.append(
            CreateBboxMaskFromEnhancedLabeld(keys=["enhanced_image"], 
                                             enhanced_label_key="enhanced_label",
                                             bbox_mask_key="bbox_mask",
                                             bbox_value=2)
        )
    
    if inpainting_mode:
        train_transforms_list.append(
            CreateInpaintMaskFromPlainLabeld(keys=["image"],
                                             plain_label_key="plain_label", 
                                             inpaint_mask_key="inpaint_mask",
                                             mask_value=2)
        )
    
    train_transforms = Compose(train_transforms_list)
    
    cache_rate = min(cache_rate, 0.5)
    
    try:
        train_ds = monai.data.CacheDataset(
            data=data_dicts, 
            transform=train_transforms, 
            cache_rate=cache_rate, 
            num_workers=min(num_workers, 2)
        )
    except Exception as e:
        print(f"CacheDataset创建失败，使用Dataset: {e}")
        train_ds = monai.data.Dataset(data=data_dicts, transform=train_transforms)
    
    return DataLoader(
        train_ds, 
        num_workers=0,
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=False,
        persistent_workers=False,
        multiprocessing_context=None
    )

def train_one_epoch_with_dual_pasta(
    epoch: int,
    unet: torch.nn.Module,
    pasta_model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    pasta_loss_weight: float = 0.1,
    amp: bool = True,
    inpainting_lambda: float = 10.0,
    unmask_lambda: float = 1.0,
    pasta_target_size: tuple = (64, 64, 64),
    pasta_padding: int = 5,
    use_dual_pasta: bool = True,
    pasta_fusion_method: str = "concat",
    enhanced_weight: float = 0.7,
    attention_fusion_layer: nn.Module = None,
) -> torch.Tensor:
    unet.train()
    loss_components = 4 if use_dual_pasta else 3
    loss_torch = torch.zeros(loss_components, dtype=torch.float, device=device)
    
    for _iter, train_data in enumerate(train_loader):
        if _iter == 0 and local_rank == 0:
            if "inpaint_mask" not in train_data:
                logger.error(f"ERROR: inpaint_mask not found in train_data! Available keys: {list(train_data.keys())}")
                mask_keys = [k for k in train_data.keys() if 'mask' in k.lower()]
                if mask_keys:
                    logger.info(f"Found mask-related keys: {mask_keys}")
                else:
                    logger.error("No mask-related keys found!")
                raise KeyError("inpaint_mask not found in training data")
        
        images = train_data["image"].to(device)
        images = images * scale_factor
        class_label = train_data["class_label"].to(device)
        spacing_tensor = train_data["spacing"].to(device)
        
        inpaint_mask = train_data["inpaint_mask"].to(device)
        
        if use_dual_pasta and pasta_model is not None:
            plain_raw_images = train_data["plain_raw_image"].to(device)
            enhanced_images = train_data["enhanced_image"].to(device)
            bbox_masks = train_data["bbox_mask"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast("cuda", enabled=amp):
            target_features = None
            if use_dual_pasta and pasta_model is not None:
                target_features = extract_dual_pasta_features(
                    pasta_model=pasta_model,
                    plain_images=plain_raw_images,
                    enhanced_images=enhanced_images,
                    bbox_masks=bbox_masks,
                    pasta_target_size=pasta_target_size,
                    pasta_padding=pasta_padding,
                    fusion_method=pasta_fusion_method,
                    enhanced_weight=enhanced_weight,
                    attention_fusion_layer=attention_fusion_layer
                )
            
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
            
            noisy_latent = images.clone()
            noisy_mask_region = noise_scheduler.add_noise(
                original_samples=images, 
                noise=noise, 
                timesteps=timesteps
            )
            noisy_latent = (1 - inpaint_mask) * images + inpaint_mask * noisy_mask_region
            
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
                "class_labels": class_label,
                "inpaint_mask": inpaint_mask,
            }
            
            if use_dual_pasta and pasta_model is not None:
                model_output, predicted_features = unet(**unet_inputs)
            else:
                model_output = unet(**unet_inputs)
                predicted_features = None
            
            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                model_gt = images - noise
            else:
                raise ValueError("Unsupported prediction type")
                
            mask_loss = F.l1_loss(model_output * inpaint_mask, model_gt * inpaint_mask)
            zero_target = torch.zeros_like(model_output)
            unmask_loss = F.l1_loss(model_output * (1 - inpaint_mask), zero_target * (1 - inpaint_mask))
            denoising_loss = inpainting_lambda * mask_loss + unmask_lambda * unmask_loss
            
            pasta_loss = torch.tensor(0.0, device=device)
            if use_dual_pasta and pasta_model is not None and target_features is not None and predicted_features is not None:
                if pasta_fusion_method == "concat":
                    if predicted_features.shape[-1] != target_features.shape[-1]:
                        logger.warning(f"Feature dimension mismatch: predicted {predicted_features.shape[-1]}, target {target_features.shape[-1]}")
                        if predicted_features.shape[-1] * 2 == target_features.shape[-1]:
                            predicted_features = torch.cat([predicted_features, predicted_features], dim=-1)
                
                norm_target_features = F.normalize(target_features, dim=-1)
                norm_predicted_features = F.normalize(predicted_features, dim=-1)
                pasta_loss = -(norm_predicted_features * norm_target_features).sum(dim=-1).mean()
            
            if use_dual_pasta and pasta_model is not None:
                loss = denoising_loss + pasta_loss_weight * pasta_loss
            else:
                loss = denoising_loss
        
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0
        loss_torch[2] += denoising_loss.item()
        if use_dual_pasta and len(loss_torch) > 3:
            loss_torch[3] += pasta_loss.item()
        
        if local_rank == 0 and _iter % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            if use_dual_pasta:
                logger.info(
                    f"[{datetime.now()}] e {epoch + 1}, iter {_iter+1}/{len(train_loader)}, "
                    f"total_loss: {loss.item():.4f}, denoise_loss: {denoising_loss.item():.4f}, "
                    f"pasta_loss: {pasta_loss.item():.4f}, lr: {current_lr:.8f}, fusion: {pasta_fusion_method}"
                )
            else:
                logger.info(
                    f"[{datetime.now()}] e {epoch + 1}, iter {_iter+1}/{len(train_loader)}, "
                    f"total_loss: {loss.item():.4f}, denoise_loss: {denoising_loss.item():.4f}, "
                    f"lr: {current_lr:.8f}, mode: inpainting_only"
                )
    
    return loss_torch

def safe_calculate_scale_factor(train_loader, device, logger):
    try:
        for batch in train_loader:
            images = batch["image"].to(device)
            scale_factor = 1.0 / torch.std(images)
            logger.info(f"Calculated scale factor: {scale_factor.item():.6f}")
            return scale_factor
    except Exception as e:
        logger.warning(f"Failed to calculate scale factor from data: {e}")
        logger.info("Using default scale factor: 1.0")
        return torch.tensor(1.0, device=device)

def diff_model_train_with_dual_pasta_fixed(
    env_config_path: str, 
    model_config_path: str, 
    model_def_path: str, 
    num_gpus: int, 
    amp: bool = True,
    inpainting_mode: bool = True,
    use_dual_pasta: bool = True,
    pasta_checkpoint_path: str = "/path/to/your/PASTA_final.pth",
    pasta_loss_weight: float = 0.1,
    pasta_target_size: tuple = (64, 64, 64),
    pasta_padding: int = 5,
    pasta_fusion_method: str = "concat",
    enhanced_weight: float = 0.7,
) -> None:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")
    
    if inpainting_mode:
        logger.info("Training in inpainting mode")
    if use_dual_pasta:
        logger.info(f"Using Dual PASTA alignment with fusion method: {pasta_fusion_method}")
        logger.info(f"Enhanced image weight: {enhanced_weight}")
        logger.info(f"PASTA target size: {pasta_target_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    pasta_model = None
    attention_fusion_layer = None
    if use_dual_pasta:
        pasta_model = load_pasta_model(pasta_checkpoint_path, device)
        
        if pasta_fusion_method == "attention":
            pasta_feature_dim = 512
            attention_fusion_layer = DualImageAttentionFusion(pasta_feature_dim).to(device)

    unet = load_unet(args, device, logger)
    
    if use_dual_pasta and pasta_fusion_method == "concat":
        original_feature_dim = 512
        target_feature_dim = original_feature_dim * 2
        
        if hasattr(unet, 'feature_projection'):
            unet.feature_projection = nn.Linear(original_feature_dim, target_feature_dim).to(device)
            logger.info(f"Added feature projection layer: {original_feature_dim} -> {target_feature_dim}")
    
    noise_scheduler = define_instance(args, "noise_scheduler")

    train_files = load_filenames(args.json_data_list)
    
    if local_rank == 0:
        logger.info(f"num_files_train: {len(train_files)}")
    
    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    train_loader = prepare_data_from_json_dual_pasta(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        args,
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=False,
        inpainting_mode=inpainting_mode,
        use_dual_pasta=use_dual_pasta,
    )
    
    logger.info(f"Batch Size: {args.diffusion_unet_train['batch_size']}")
    
    scale_factor = safe_calculate_scale_factor(train_loader, device, logger)
    
    unet_params = list(unet.parameters())
    if attention_fusion_layer is not None:
        unet_params.extend(list(attention_fusion_layer.parameters()))
    
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    if attention_fusion_layer is not None:
        optimizer = torch.optim.AdamW(unet_params, lr=args.diffusion_unet_train["lr"])

    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch_with_dual_pasta(
            epoch=epoch,
            unet=unet,
            pasta_model=pasta_model,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            scale_factor=scale_factor,
            noise_scheduler=noise_scheduler,
            num_train_timesteps=args.noise_scheduler["num_train_timesteps"],
            device=device,
            logger=logger,
            local_rank=local_rank,
            pasta_loss_weight=pasta_loss_weight,
            amp=amp,
            inpainting_lambda=10.0,
            unmask_lambda=5.0,
            pasta_target_size=pasta_target_size,
            pasta_padding=pasta_padding,
            use_dual_pasta=use_dual_pasta,
            pasta_fusion_method=pasta_fusion_method,
            enhanced_weight=enhanced_weight,
            attention_fusion_layer=attention_fusion_layer,
        )
        
        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            total_loss_epoch = loss_torch[0] / loss_torch[1]
            denoising_loss_epoch = loss_torch[2] / loss_torch[1]
            
            logger.info(f"epoch {epoch + 1} average total loss: {total_loss_epoch:.4f}")
            logger.info(f"  └─ average denoising loss: {denoising_loss_epoch:.4f}")
            
            if use_dual_pasta and len(loss_torch) > 3:
                pasta_loss_epoch = loss_torch[3] / loss_torch[1]
                logger.info(f"  └─ average dual pasta loss: {pasta_loss_epoch:.4f}")
                logger.info(f"  └─ fusion method: {pasta_fusion_method}")
            else:
                logger.info(f"  └─ Dual PASTA alignment: disabled")

            save_checkpoint(
                epoch, unet, total_loss_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor, args.model_dir, args
            )

    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
) -> None:
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
        },
        f"{ckpt_folder}/{args.model_filename}",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Dual PASTA Diffusion Model Training")
    parser.add_argument("--env_config", type=str, default="./configs/environment_maisi_diff_model.json")
    parser.add_argument("--model_config", type=str, default="./configs/config_maisi_diff_model.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow_PASTA_no_mask.json")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    
    parser.add_argument("--use_dual_pasta", action="store_true")
    parser.add_argument("--pasta_ckpt", type=str, default=os.getenv('PASTA_CHECKPOINT_PATH', './models/PASTA_final.pth'))
    parser.add_argument("--pasta_weight", type=float, default=0.1)
    parser.add_argument("--pasta_fusion_method", type=str, default="concat", choices=["concat", "add", "weighted", "attention"])
    parser.add_argument("--enhanced_weight", type=float, default=0.7)
    parser.add_argument("--pasta_target_size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--pasta_padding", type=int, default=5)
    
    args = parser.parse_args()
    
    diff_model_train_with_dual_pasta_fixed(
        args.env_config, 
        args.model_config, 
        args.model_def, 
        args.num_gpus, 
        args.amp,
        use_dual_pasta=args.use_dual_pasta,
        pasta_checkpoint_path=args.pasta_ckpt,
        pasta_loss_weight=args.pasta_weight,
        pasta_fusion_method=args.pasta_fusion_method,
        enhanced_weight=args.enhanced_weight,
        pasta_target_size=tuple(args.pasta_target_size),
        pasta_padding=args.pasta_padding
    )

