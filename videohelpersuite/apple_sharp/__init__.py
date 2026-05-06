"""SHARP model: params, config presets, and factory functions.

Consolidates params.py, presets/, and all create_* factories into one file.
Exports PredictorParams and create_predictor for use by load_model.py.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

from .color_space import ColorSpace

from .model import (
    DirectPredictionHead,
    GaussianComposer,
    GaussianDensePredictionTransformer,
    LearnedAlignment,
    MonodepthDensePredictionTransformer,
    MonodepthWithEncodingAdaptor,
    MultiLayerInitializer,
    MultiresConvDecoder,
    NormLayerName,
    RGBGaussianPredictor,
    SlidingPyramidNetwork,
    UpsamplingMode,
    VisionTransformer,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ViT Presets
# ---------------------------------------------------------------------------

ViTPreset = Literal["dinov2l16_384"]
MLPMode = Literal["vanilla", "glu"]


@dataclasses.dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int
    init_values: float

    img_size: int = 384
    patch_size: int = 16
    num_classes: int = 21841
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    global_pool: str = "avg"
    mlp_mode: MLPMode = "vanilla"
    intermediate_features_ids: list[int] | None = None


VIT_CONFIG_DICT: dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        global_pool="",
    ),
}

# ---------------------------------------------------------------------------
# Monodepth Presets
# ---------------------------------------------------------------------------

MONODEPTH_ENCODER_DIMS_MAP: dict[ViTPreset, list[int]] = {
    "dinov2l16_384": [256, 512, 1024, 1024],
}

MONODEPTH_HOOK_IDS_MAP: dict[ViTPreset, list[int]] = {
    "dinov2l16_384": [5, 11, 17, 23],
}

# ---------------------------------------------------------------------------
# Parameter Dataclasses
# ---------------------------------------------------------------------------

DimsDecoder = tuple[int, int, int, int, int]
DPTImageEncoderType = Literal["skip_conv", "skip_conv_kernel2"]
ColorInitOption = Literal["none", "first_layer", "all_layers"]
DepthInitOption = Literal["surface_min", "surface_max", "base_depth", "linear_disparity"]


@dataclasses.dataclass
class AlignmentParams:
    """Parameters for depth alignment."""

    kernel_size: int = 16
    stride: int = 1
    frozen: bool = False
    steps: int = 4
    activation_type: str = "exp"
    depth_decoder_features: bool = False
    base_width: int = 16


@dataclasses.dataclass
class DeltaFactor:
    """Factors to multiply deltas with before activation."""

    xy: float = 0.001
    z: float = 0.001
    color: float = 0.1
    opacity: float = 1.0
    scale: float = 1.0
    quaternion: float = 1.0


@dataclasses.dataclass
class InitializerParams:
    """Parameters for initializer."""

    scale_factor: float = 1.0
    disparity_factor: float = 1.0
    stride: int = 2
    num_layers: int = 2
    first_layer_depth_option: DepthInitOption = "surface_min"
    rest_layer_depth_option: DepthInitOption = "surface_min"
    color_option: ColorInitOption = "all_layers"
    base_depth: float = 10.0
    feature_input_stop_grad: bool = False
    normalize_depth: bool = True
    output_inpainted_layer_only: bool = False
    set_uninpainted_opacity_to_zero: bool = False
    concat_inpainting_mask: bool = False


@dataclasses.dataclass
class MonodepthParams:
    """Parameters for monodepth network."""

    patch_encoder_preset: ViTPreset = "dinov2l16_384"
    image_encoder_preset: ViTPreset = "dinov2l16_384"
    checkpoint_uri: str | None = None
    unfreeze_patch_encoder: bool = False
    unfreeze_image_encoder: bool = False
    unfreeze_decoder: bool = False
    unfreeze_head: bool = False
    unfreeze_norm_layers: bool = False
    grad_checkpointing: bool = False
    use_patch_overlap: bool = True
    dims_decoder: DimsDecoder = (256, 256, 256, 256, 256)


@dataclasses.dataclass
class MonodepthAdaptorParams:
    """Parameters for monodepth network feature adaptor."""

    encoder_features: bool = True
    decoder_features: bool = False


@dataclasses.dataclass
class GaussianDecoderParams:
    """Parameters for backbone with default values."""

    dim_in: int = 5
    dim_out: int = 32
    norm_type: NormLayerName = "group_norm"
    norm_num_groups: int = 8
    stride: int = 2
    patch_encoder_preset: ViTPreset = "dinov2l16_384"
    image_encoder_preset: ViTPreset = "dinov2l16_384"
    dims_decoder: DimsDecoder = (128, 128, 128, 128, 128)
    use_depth_input: bool = True
    grad_checkpointing: bool = False
    upsampling_mode: UpsamplingMode = "transposed_conv"
    image_encoder_type: DPTImageEncoderType = "skip_conv_kernel2"


@dataclasses.dataclass
class PredictorParams:
    """Parameters for predictors with default values."""

    initializer: InitializerParams = dataclasses.field(default_factory=InitializerParams)
    monodepth: MonodepthParams = dataclasses.field(default_factory=MonodepthParams)
    monodepth_adaptor: MonodepthAdaptorParams = dataclasses.field(
        default_factory=MonodepthAdaptorParams
    )
    gaussian_decoder: GaussianDecoderParams = dataclasses.field(
        default_factory=GaussianDecoderParams
    )
    depth_alignment: AlignmentParams = dataclasses.field(default_factory=AlignmentParams)
    delta_factor: DeltaFactor = dataclasses.field(default_factory=DeltaFactor)
    max_scale: float = 10.0
    min_scale: float = 0.0
    norm_type: NormLayerName = "group_norm"
    norm_num_groups: int = 8
    use_predicted_mean: bool = False
    color_activation_type: str = "sigmoid"
    opacity_activation_type: str = "sigmoid"
    color_space: ColorSpace = "linearRGB"
    low_pass_filter_eps: float = 1e-2
    num_monodepth_layers: int = 2
    sorting_monodepth: bool = False
    base_scale_on_predicted_mean: bool = True


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_vit(
    config: ViTConfig | None = None,
    preset: ViTPreset | None = "dinov2l16_384",
    intermediate_features_ids: list[int] | None = None,
    dtype=None,
    device=None,
    operations=None,
) -> VisionTransformer:
    """Factory function for creating a ViT model."""
    if config is not None:
        LOGGER.info("Using user-defined config.")
    else:
        if preset is None:
            raise ValueError("User-defined config and preset cannot be both None.")
        LOGGER.info("Using preset ViT %s.", preset)
        config = VIT_CONFIG_DICT[preset]

    config.intermediate_features_ids = intermediate_features_ids

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        init_values=config.init_values,
        use_glu_mlp=(config.mlp_mode == "glu"),
        num_classes=config.num_classes,
        intermediate_features_ids=config.intermediate_features_ids,
        dtype=dtype,
        device=device,
        operations=operations,
    )
    return model


def create_monodepth_encoder(
    patch_encoder_preset: ViTPreset,
    image_encoder_preset: ViTPreset,
    use_patch_overlap: bool = True,
    last_encoder: int = 256,
    dtype=None,
    device=None,
    operations=None,
) -> SlidingPyramidNetwork:
    """Create the SPN-based monodepth encoder."""
    dims_encoder = [last_encoder] + MONODEPTH_ENCODER_DIMS_MAP[patch_encoder_preset]
    patch_encoder_block_ids = MONODEPTH_HOOK_IDS_MAP[patch_encoder_preset]

    patch_encoder = create_vit(
        preset=patch_encoder_preset,
        intermediate_features_ids=patch_encoder_block_ids,
        dtype=dtype, device=device, operations=operations,
    )
    image_encoder = create_vit(
        preset=image_encoder_preset,
        intermediate_features_ids=None,
        dtype=dtype, device=device, operations=operations,
    )

    return SlidingPyramidNetwork(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        use_patch_overlap=use_patch_overlap,
        dtype=dtype, device=device, operations=operations,
    )


def create_monodepth_decoder(
    patch_encoder_preset: ViTPreset,
    dims_decoder=None,
    dtype=None,
    device=None,
    operations=None,
) -> MultiresConvDecoder:
    """Create monodepth decoder."""
    dims_encoder = MONODEPTH_ENCODER_DIMS_MAP[patch_encoder_preset]
    if dims_decoder is None:
        dims_decoder = dims_encoder[0]
    if isinstance(dims_decoder, int):
        dims_decoder = [dims_decoder]
    return MultiresConvDecoder(
        dims_encoder=[dims_decoder[0]] + list(dims_encoder),
        dims_decoder=dims_decoder,
        dtype=dtype, device=device, operations=operations,
    )


def create_monodepth_dpt(
    params: MonodepthParams | None = None,
    dtype=None,
    device=None,
    operations=None,
) -> MonodepthDensePredictionTransformer:
    """Create MonodepthDensePredictionTransformer model."""
    if params is None:
        params = MonodepthParams()

    encoder = create_monodepth_encoder(
        params.patch_encoder_preset,
        params.image_encoder_preset,
        use_patch_overlap=params.use_patch_overlap,
        last_encoder=params.dims_decoder[0],
        dtype=dtype, device=device, operations=operations,
    )
    decoder = create_monodepth_decoder(
        params.patch_encoder_preset, params.dims_decoder,
        dtype=dtype, device=device, operations=operations,
    )

    return MonodepthDensePredictionTransformer(
        encoder=encoder, decoder=decoder, last_dims=(32, 1),
        dtype=dtype, device=device, operations=operations,
    )


def create_monodepth_adaptor(
    monodepth_predictor: MonodepthDensePredictionTransformer,
    params: MonodepthAdaptorParams,
    num_monodepth_layers: int,
    sorting_monodepth: bool,
) -> MonodepthWithEncodingAdaptor:
    """Create an adaptor that returns both disparity and features."""
    return MonodepthWithEncodingAdaptor(
        monodepth_predictor=monodepth_predictor,
        return_encoder_features=params.encoder_features,
        return_decoder_features=params.decoder_features,
        num_monodepth_layers=num_monodepth_layers,
        sorting_monodepth=sorting_monodepth,
    )


def create_gaussian_decoder(
    params: GaussianDecoderParams,
    dims_depth_features: list[int],
    dtype=None,
    device=None,
    operations=None,
) -> GaussianDensePredictionTransformer:
    """Create gaussian_decoder model."""
    decoder = MultiresConvDecoder(
        dims_depth_features,
        params.dims_decoder,
        upsampling_mode=params.upsampling_mode,
        dtype=dtype, device=device, operations=operations,
    )

    return GaussianDensePredictionTransformer(
        decoder=decoder,
        dim_in=params.dim_in,
        dim_out=params.dim_out,
        stride_out=params.stride,
        norm_type=params.norm_type,
        norm_num_groups=params.norm_num_groups,
        use_depth_input=params.use_depth_input,
        image_encoder_type=params.image_encoder_type,
        image_encoder_params=params,
        dtype=dtype, device=device, operations=operations,
    )


def create_initializer(params: InitializerParams) -> MultiLayerInitializer:
    """Create initializer."""
    return MultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
    )


def create_alignment(
    params: AlignmentParams,
    depth_decoder_dim: int | None = None,
    dtype=None,
    device=None,
    operations=None,
) -> LearnedAlignment | None:
    """Create depth alignment."""
    if depth_decoder_dim is None:
        raise ValueError("Requires depth_decoder_dim for LearnedAlignment.")
    return LearnedAlignment(
        depth_decoder_features=params.depth_decoder_features,
        depth_decoder_dim=depth_decoder_dim,
        steps=params.steps,
        stride=params.stride,
        base_width=params.base_width,
        activation_type=params.activation_type,
        dtype=dtype, device=device, operations=operations,
    )


def create_predictor(
    params: PredictorParams,
    dtype=None,
    device=None,
    operations=None,
) -> RGBGaussianPredictor:
    """Create gaussian predictor model.

    This is the top-level factory called by load_model.py.
    Accepts dtype, device, operations for ComfyUI-native weight management.
    """
    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError(
            "We donot expected gaussian_decoder has higher resolution than initializer."
        )

    scale_factor = params.gaussian_decoder.stride // params.initializer.stride
    gaussian_composer = GaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )
    if params.num_monodepth_layers > 1 and params.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model = create_monodepth_dpt(
        params.monodepth,
        dtype=dtype, device=device, operations=operations,
    )
    monodepth_adaptor = create_monodepth_adaptor(
        monodepth_model,
        params.monodepth_adaptor,
        params.num_monodepth_layers,
        params.sorting_monodepth,
    )

    if params.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(params.num_monodepth_layers)

    gaussian_decoder = create_gaussian_decoder(
        params.gaussian_decoder,
        dims_depth_features=monodepth_adaptor.get_feature_dims(),
        dtype=dtype, device=device, operations=operations,
    )
    initializer = create_initializer(params.initializer)
    prediction_head = DirectPredictionHead(
        feature_dim=gaussian_decoder.dim_out, num_layers=initializer.num_layers,
        dtype=dtype, device=device, operations=operations,
    )
    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return RGBGaussianPredictor(
        init_model=initializer,
        feature_model=gaussian_decoder,
        prediction_head=prediction_head,
        monodepth_model=monodepth_adaptor,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=create_alignment(
            params.depth_alignment, depth_decoder_dim=decoder_dim,
            dtype=dtype, device=device, operations=operations,
        ),
        dtype=dtype,
    )


__all__ = [
    "PredictorParams",
    "create_predictor",
]
