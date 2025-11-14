
from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime
from icecream import ic
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import sliding_window_inference
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from monai.utils import set_determinism
from tqdm import tqdm
import monai
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .sample import ReconModel, check_input
from .utils import define_instance, dynamic_infer


def set_random_seed(seed: int) -> int:
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    ic(f"{args.model_dir}/{args.model_filename}")
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device: torch.device) -> tuple:
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones(
        (len(spacing_tensor)), dtype=torch.long
    ).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor


def encode_image(
    image: torch.Tensor, 
    autoencoder: torch.nn.Module, 
    device: torch.device, 
    logger: logging.Logger
) -> torch.Tensor:
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            if image.dim() == 4:
                image = image.unsqueeze(0)
            
            latent = autoencoder.encode_stage_2_inputs(image)
            logger.info(f"Encoded latent shape: {latent.shape}")
            
            return latent


def downsample_mask(
    mask: torch.Tensor, 
    latent_shape: torch.Size, 
    device: torch.device, 
    logger: logging.Logger
) -> torch.Tensor:
    logger.info(f"Original mask shape: {mask.shape}, latent shape: {latent_shape}")
    
    if mask.dim() == 4:
        mask = mask.unsqueeze(0)
    
    if len(latent_shape) == 5:
        target_size = latent_shape[2:]
    elif len(latent_shape) == 4:
        target_size = latent_shape[1:]
    else:
        target_size = latent_shape
    
    logger.info(f"Target spatial size: {target_size}")
    
    import torch.nn.functional as F
    downsampled_mask = F.interpolate(
        mask.float(),
        size=tuple(target_size),
        mode='nearest'
    )
    
    logger.info(f"Downsampled mask shape: {downsampled_mask.shape}")
    
    return downsampled_mask


def load_image_and_mask(
    image_path: str, 
    mask_path: str, 
    device: torch.device, 
    logger: logging.Logger
) -> tuple:
    img_nib = nib.load(image_path)
    original_affine = img_nib.affine
    logger.info(f"原始图像仿射矩阵:\n{original_affine}")
    
    transforms = Compose([
        monai.transforms.LoadImaged(keys=["image", "mask"]),
        monai.transforms.EnsureChannelFirstd(keys=["image", "mask"]),
        
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        monai.transforms.ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
        ),
    ])

    data = transforms({"image": image_path, "mask": mask_path})
    image = data["image"].to(device)
    
    inpaint_mask = (data["mask"] == 2).float().to(device)
    
    logger.info(f"已加载图像: {image.shape}, 掩码: {inpaint_mask.shape}")
    return image, inpaint_mask, original_affine


def run_inference_with_inpainting(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    image_path: str,
    mask_path: str,
    logger: logging.Logger,
) -> tuple:
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    inpainting_mode = hasattr(unet, 'inpainting_mode') and unet.inpainting_mode

    original_image, original_mask, original_affine = load_image_and_mask(
        image_path, mask_path, device, logger
    )
    
    with torch.no_grad():
        latent = encode_image(original_image, autoencoder, device, logger)
        latent = latent * scale_factor
    
    inpaint_mask = downsample_mask(original_mask, latent.shape, device, logger)
    
    noise = torch.randn_like(latent, device=device)
    
    logger.info(f"潜在变量: {latent.shape}, 掩码: {inpaint_mask.shape}, 噪声: {noise.shape}")

    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(latent.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )
    
    first_timestep = all_timesteps[0]
    t_tensor = torch.tensor([first_timestep], device=device)
    
    noised_latent = noise_scheduler.add_noise(original_samples=latent, noise=noise, timesteps=t_tensor)
    image = latent * (1.0 - inpaint_mask) + noised_latent * inpaint_mask

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }
            
            if include_body_region:
                unet_inputs.update({
                    "top_region_index_tensor": top_region_index_tensor,
                    "bottom_region_index_tensor": bottom_region_index_tensor,
                })

            if include_modality:
                unet_inputs.update({
                    "class_labels": modality_tensor,
                })
                
            if inpainting_mode:
                unet_inputs.update({
                    "inpaint_mask": inpaint_mask
                })

            model_output,_ = unet(**unet_inputs)
            
            if not isinstance(noise_scheduler, RFlowScheduler):
                new_image, _ = noise_scheduler.step(model_output, t, image)
            else:
                new_image, _ = noise_scheduler.step(model_output, t, image, next_t)
            
            image = latent * (1.0 - inpaint_mask) + new_image * inpaint_mask

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
        data = synthetic_images.squeeze().cpu().detach().numpy()
        
        a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
        
        return np.int16(data), original_affine


def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    original_affine: np.ndarray,
    logger: logging.Logger,
) -> None:
    out_affine = original_affine.copy()
    
    for i in range(3):
        current_spacing = np.sqrt(np.sum(original_affine[:3, i]**2))
        if current_spacing > 0:
            scale_factor = out_spacing[i] / current_spacing
            out_affine[:3, i] = original_affine[:3, i] * scale_factor

    logger.info(f"输出仿射矩阵:\n{out_affine}")
    
    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"已保存 {output_path}.")


@torch.inference_mode()
def diff_model_inpaint(
    env_config_path: str, 
    model_config_path: str, 
    model_def_path: str, 
    image_path: str,
    mask_path: str,
    num_gpus: int
) -> None:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inpainting")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"使用 {device} (共 {world_size} 个设备)，随机种子: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    output_prefix = args.output_prefix
    ckpt_filepath = f"{args.model_dir}/{args.model_filename}"

    if local_rank == 0:
        logger.info(f"[配置] ckpt_filepath -> {ckpt_filepath}.")
        logger.info(f"[配置] random_seed -> {random_seed}.")
        logger.info(f"[配置] output_prefix -> {output_prefix}.")
        logger.info(f"[配置] output_size -> {output_size}.")
        logger.info(f"[配置] out_spacing -> {out_spacing}.")
        logger.info(f"[配置] image_path -> {image_path}.")
        logger.info(f"[配置] mask_path -> {mask_path}.")

    if not os.path.exists(image_path):
        logger.error(f"找不到图像文件: {image_path}")
        return
    
    if not os.path.exists(mask_path):
        logger.error(f"找不到掩码文件: {mask_path}")
        return

    check_input(None, None, None, output_size, out_spacing, None)

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)
    
    data, original_affine = run_inference_with_inpainting(
        args,
        device,
        autoencoder,
        unet,
        scale_factor,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
        image_path,
        mask_path,
        logger,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = "{0}/{1}_inpainted_seed{2}_size{3:d}x{4:d}x{5:d}_{6}_rank{7}.nii.gz".format(
        args.output_dir,
        output_prefix,
        random_seed,
        output_size[0],
        output_size[1],
        output_size[2],
        timestamp,
        local_rank,
    )
    
    save_image(data, output_size, out_spacing, output_path, original_affine, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inpainting")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi3d-rflow_PASTA_no_mask.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./dataset/cropped_192_kidney_plain_only/000_plain_sample_0/cropped_image.nii.gz",
        help="Path to the original image file",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./dataset/cropped_192_kidney_plain_only/000_plain_sample_0/cropped_mask.nii.gz",
        help="Path to the mask file (value 2 indicates regions to inpaint)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )

    args = parser.parse_args()
    diff_model_inpaint(
        args.env_config, 
        args.model_config, 
        args.model_def, 
        args.image_path,
        args.mask_path,
        args.num_gpus
    )
