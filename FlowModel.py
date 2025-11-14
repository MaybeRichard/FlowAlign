from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
from torch import nn

from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import (
    get_down_block,
    get_mid_block,
    get_timestep_embedding,
    get_up_block,
    zero_module,
)
from monai.utils import ensure_tuple_rep
from monai.utils.type_conversion import convert_to_tensor

__all__ = ["FlowModel"]

class FlowModel(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        include_spacing_input: bool = False,
        inpainting_mode: bool = True,
        with_pasta_projection: bool = True, 
        pasta_projector_hidden_dim: int = 2048,
        pasta_target_dim: int = 1024,
        multi_scale_projection: bool = False,
        multi_scale_levels: tuple = (1, 2, 3),
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "FlowModel expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "FlowModel expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"FlowModel expects all num_channels being multiple of norm_num_groups, "
                f"but get num_channels: {num_channels} and norm_num_groups: {norm_num_groups}"
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                f"FlowModel expects num_channels being same size of attention_levels, "
                f"but get num_channels: {len(num_channels)} and attention_levels: {len(attention_levels)}"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )
        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.inpainting_mode = inpainting_mode
        self.with_pasta_projection = with_pasta_projection
        self.multi_scale_projection = multi_scale_projection
        self.multi_scale_levels = multi_scale_levels

        actual_in_channels = in_channels * 2 if inpainting_mode else in_channels
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=actual_in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        time_embed_dim = num_channels[0] * 4
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        self.include_top_region_index_input = include_top_region_index_input
        self.include_bottom_region_index_input = include_bottom_region_index_input
        self.include_spacing_input = include_spacing_input

        new_time_embed_dim = time_embed_dim
        if self.include_top_region_index_input:
            self.top_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_bottom_region_index_input:
            self.bottom_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_spacing_input:
            self.spacing_layer = self._create_embedding_module(3, time_embed_dim)
            new_time_embed_dim += time_embed_dim

        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.down_blocks.append(down_block)

        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
        if self.with_pasta_projection:
            if self.multi_scale_projection:
                self.pasta_projectors = nn.ModuleList()
                total_projected_channels = 0

                for level in self.multi_scale_levels:
                    if level < len(num_channels):
                        level_channels = num_channels[level]
                        self.pasta_projectors.append(nn.AdaptiveAvgPool3d(1))
                        total_projected_channels += level_channels
                
                self.pasta_projectors.append(nn.AdaptiveAvgPool3d(1))
                total_projected_channels += num_channels[-1]

                self.final_pasta_projector = nn.Sequential(
                    nn.Linear(total_projected_channels, pasta_projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(pasta_projector_hidden_dim, pasta_target_dim),
                )

            else:
                bottleneck_channels = num_channels[-1]
                self.pasta_pool = nn.AdaptiveAvgPool3d(1)
                self.pasta_projector = nn.Sequential(
                    nn.Linear(bottleneck_channels, pasta_projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(pasta_projector_hidden_dim, pasta_projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(pasta_projector_hidden_dim, pasta_target_dim),
                )
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.up_blocks.append(up_block)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model

    def _get_time_and_class_embedding(self, x, timesteps, class_labels):
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb += class_emb
        return emb

    def _get_input_embeddings(self, emb, top_index, bottom_index, spacing):
        if self.include_top_region_index_input:
            _emb = self.top_region_index_layer(top_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_bottom_region_index_input:
            _emb = self.bottom_region_index_layer(bottom_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_spacing_input:
            _emb = self.spacing_layer(spacing)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        multi_scale_features: List[torch.Tensor] = []
        for i, downsample_block in enumerate(self.down_blocks):
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)
            if self.with_pasta_projection and self.multi_scale_projection and i in self.multi_scale_levels:
                multi_scale_features.append(h)


        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples.append(down_block_res_sample)

            down_block_res_samples = new_down_block_res_samples
        return h, down_block_res_samples, multi_scale_features

    def _apply_up_blocks(self, h, emb, context, down_block_res_samples):
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

        return h

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
        inpaint_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.inpainting_mode:
            if inpaint_mask is None:
                inpaint_mask = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
            
            if inpaint_mask.shape[1] == 1 and x.shape[1] > 1:
                inpaint_mask = inpaint_mask.expand(-1, x.shape[1], -1, -1, -1)
            
            x_combined = torch.cat([x, inpaint_mask], dim=1)
        else:
            x_combined = x

        emb = self._get_time_and_class_embedding(x_combined, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
        
        h = self.conv_in(x_combined)
        
        h, _updated_down_block_res_samples, multi_scale_encoder_features = self._apply_down_blocks(
            h, emb, context, down_block_additional_residuals
        )
        
        h_bottleneck = self.middle_block(h, emb, context)

        if mid_block_additional_residual is not None:
            h_bottleneck += mid_block_additional_residual

        predicted_features = None
        if self.with_pasta_projection:
            if self.multi_scale_projection:
                predicted_features_list = []
                
                for feature, projector in zip(multi_scale_encoder_features, self.pasta_projectors):
                    pooled = projector(feature)
                    predicted_features_list.append(torch.flatten(pooled, 1))

                pooled_bottleneck = self.pasta_projectors[-1](h_bottleneck)
                predicted_features_list.append(torch.flatten(pooled_bottleneck, 1))
                
                combined_features = torch.cat(predicted_features_list, dim=1)
                
                predicted_features = self.final_pasta_projector(combined_features)

            else:
                pooled_features = self.pasta_pool(h_bottleneck)
                pooled_features = torch.flatten(pooled_features, 1)
                predicted_features = self.pasta_projector(pooled_features)

        h = self._apply_up_blocks(h_bottleneck, emb, context, _updated_down_block_res_samples)
        
        denoising_output = self.out(h)
        denoising_output = convert_to_tensor(denoising_output)
        
        return denoising_output, predicted_features
