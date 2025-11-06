from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from vanilla_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, UncertaintyRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead, UncertaintyFieldHead



@dataclass
class ActiveNeRFModelConfig(ModelConfig):
    """ActiveNeRF Model Config - COMPLETO"""

    _target: Type = field(default_factory=lambda: ActiveNeRFModel)
    
    # ===== PARÁMETROS QUE FALTABAN =====
    beta_min: float = 0.01
    """Minimum uncertainty value"""
    
    w: float = 0.01
    """Weight for uncertainty loss"""
    
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    
    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    
    temporal_distortion_params: Dict[str, Any] = field(
        default_factory=lambda: {"kind": "dnerf"}
    )
    """Parameters to instantiate temporal distortion with"""
    
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    
    # ===== PARÁMETROS DE ACTIVENERF =====
    use_uncertainty: bool = True
    """Enable uncertainty estimation (CRITICAL for ActiveNeRF)"""
    
    enable_uncertainty_estimation: bool = True
    """Enable additional uncertainty estimation mechanisms"""
    
    uncertainty_method: str = "density_variance"
    """Method for computing uncertainty"""
    
    uncertainty_loss_weight: float = 0.01
    """Weight for uncertainty regularization loss"""
    
    use_uncertainty_loss: bool = True
    """Whether to include uncertainty in the loss function"""
    
    enable_entropy_regularization: bool = True
    """Encourage exploration through entropy regularization"""
    
    entropy_loss_weight: float = 0.001
    """Weight for entropy regularization"""
    
    loss_coefficients: Dict[str, float] = field(default_factory=lambda: {
        "rgb_loss_coarse": 1.0,
        "rgb_loss_fine": 1.0,
        "uncertainty_loss": 0.01,
        "entropy_loss": 0.001,
    })
    """Loss coefficients for different loss terms"""



# ===== MODELO =====
class ActiveNeRFModel(Model):
    """
    ActiveNeRF Model with Uncertainty Estimation
    
    Args:
        config: ActiveNeRF configuration to instantiate model
    """

    config: ActiveNeRFModelConfig

    def __init__(
        self,
        config: ActiveNeRFModelConfig,
        **kwargs,
    ) -> None:
        # Usar parámetros del config (no del __init__)
        self.beta_min = config.beta_min
        self.w = config.w
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None
        self.out_dim = None
        
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # ===== Encodings =====
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # ===== Fields =====
        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            field_heads=(RGBFieldHead, UncertaintyFieldHead)
        )

        # ===== Samplers =====
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # ===== Renderers =====
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_unct = UncertaintyRenderer()

        # ===== Losses =====
        self.rgb_loss = MSELoss()

        # ===== Metrics =====
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # ===== Temporal distortion =====
        if getattr(self.config, "enable_temporal_distortion", False):
            from nerfstudio.model_components.temporal_distortions import TemporalDistortionKind
            params = self.config.temporal_distortion_params.copy()
            kind = params.pop("kind", "dnerf")
            kind_enum = TemporalDistortionKind(kind)
            self.temporal_distortion = kind_enum.to_temporal_distortion(params)
        
        # ===== Colormaps =====
        self.color_opt = colormaps.ColormapOptions(colormap='magma')

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups for optimization"""
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Forward pass returning RGB and uncertainty"""
        
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # ===== Uniform sampling =====
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # ===== Coarse field =====
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # ===== PDF sampling =====
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # ===== Fine field =====
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        
        # ===== Uncertainty =====
        uncert_field = weights_fine + self.beta_min #field_outputs_fine[FieldHeadNames.UNCERTAINTY] + self.beta_min
        uncert_fine = self.renderer_unct(betas=uncert_field, weights=weights_fine)
        
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "uncertainty_fine": uncert_fine,
            "weights_fine": weights_fine,  # Para entropy loss
            "density_fine": field_outputs_fine[FieldHeadNames.DENSITY],  # Para análisis
        }
        return outputs
    
    def get_uncertainty(self, ray_bundle: RayBundle):

        # ===== Uniform sampling =====
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # ===== Coarse field =====
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])

        # ===== PDF sampling =====
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)


        # ===== Fine field =====
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        # ===== Uncertainty =====
        uncert_field = weights_fine + self.beta_min #field_outputs_fine[FieldHeadNames.UNCERTAINTY] + self.beta_min
        uncert_fine = self.renderer_unct(betas=uncert_field, weights=weights_fine)
        prob = torch.sigmoid(uncert_fine)
        entropy = -(prob * torch.log(prob + 1e-10) + (1 - prob) * torch.log(1 - prob + 1e-10))
        uncertainty = entropy.mean().item()
        del ray_samples_uniform, ray_samples_pdf
        del field_outputs_coarse, field_outputs_fine
        del weights_coarse, weights_fine, uncert_field
        del uncert_fine, prob, entropy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outputs = {
            "uncertainty_fine": uncertainty
        }
        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Compute losses including uncertainty regularization"""
        
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        
        # ===== RGB Losses =====
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine
        }
        
        # ===== Uncertainty Loss (regularización) =====
        if self.config.use_uncertainty and self.config.use_uncertainty_loss:
            if "uncertainty_fine" in outputs:
                # Penalizar uncertainty muy alta (evita colapso)
                uncertainty_loss = outputs["uncertainty_fine"].mean()
                loss_dict["uncertainty_loss"] = uncertainty_loss * self.config.uncertainty_loss_weight
        
        # ===== Entropy Loss (fomentar exploración) =====
        if self.config.enable_entropy_regularization and "weights_fine" in outputs:
            weights = outputs["weights_fine"]
            # Normalizar pesos para que sumen 1
            weights_norm = weights / (weights.sum(-1, keepdim=True) + 1e-10)
            # Entropía: -sum(p * log(p))
            entropy = -(weights_norm * torch.log(weights_norm + 1e-10)).sum(-1).mean()
            # Queremos maximizar entropía (minimizar -entropy)
            loss_dict["entropy_loss"] = -entropy * self.config.entropy_loss_weight
        
        # Aplicar coeficientes
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute metrics and create visualization images"""
        
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        
        # Depth
        if self.config.collider_params is not None:
            depth_coarse = colormaps.apply_depth_colormap(
                outputs["depth_coarse"],
                accumulation=outputs["accumulation_coarse"],
                near_plane=self.config.collider_params["near_plane"],
                far_plane=self.config.collider_params["far_plane"],
            )
            depth_fine = colormaps.apply_depth_colormap(
                outputs["depth_fine"],
                accumulation=outputs["accumulation_fine"],
                near_plane=self.config.collider_params["near_plane"],
                far_plane=self.config.collider_params["far_plane"],
            )
        else:
            depth_coarse = colormaps.apply_colormap(outputs["depth_coarse"])
            depth_fine = colormaps.apply_colormap(outputs["depth_fine"])
        
        # Uncertainty
        uncertainty_fine = colormaps.apply_colormap(
            outputs["uncertainty_fine"], 
            colormap_options=self.color_opt
        )
        
        # Combine images
        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Metrics
        image_metric = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse_metric = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine_metric = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image_metric, rgb_coarse_metric)
        fine_psnr = self.psnr(image_metric, rgb_fine_metric)
        fine_ssim = self.ssim(image_metric, rgb_fine_metric)
        fine_lpips = self.lpips(image_metric, rgb_fine_metric)
        
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        
        # Añadir métrica de uncertainty
        if "uncertainty_fine" in outputs:
            avg_uncertainty = outputs["uncertainty_fine"].mean().item()
            metrics_dict["avg_uncertainty"] = float(avg_uncertainty)
        
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "uncertainty": uncertainty_fine
        }
        
        return metrics_dict, images_dict