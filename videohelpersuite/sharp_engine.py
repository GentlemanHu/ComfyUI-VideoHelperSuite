"""Independent Apple SHARP engine for VideoHelperSuite video nodes."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import folder_paths


SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"
INTERNAL_SIZE = 1536

_model_patcher = None
_model_config = None
_encode_cache: dict[str, Any] = {
    "image_hash": None,
    "monodepth_output": None,
    "image_resized": None,
}


@dataclass
class SharpScene:
    gaussians: Any
    focal_px: float
    source_size: tuple[int, int]
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    center: torch.Tensor
    radius: float
    ply_path: str


def models_dir() -> str:
    root = os.path.join(folder_paths.models_dir, "sharp")
    os.makedirs(root, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("sharp", root)
    except Exception:
        pass
    return root


def model_config(precision: str = "auto") -> dict[str, str]:
    import comfy.model_management

    load_device = comfy.model_management.get_torch_device()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision == "auto":
        if comfy.model_management.should_use_bf16(load_device):
            dtype = torch.bfloat16
        elif comfy.model_management.should_use_fp16(load_device):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = dtype_map[str(precision)]
    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    path = hf_hub_download(
        repo_id=SHARP_REPO_ID,
        filename=SHARP_FILENAME,
        local_dir=models_dir(),
    )
    return {"model_path": path, "precision": precision, "dtype": dtype_str}


def load_model(precision: str = "auto"):
    global _model_patcher, _model_config
    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils

    from . import apple_sharp
    cfg = model_config(precision)
    if _model_patcher is not None and _model_config == cfg:
        return _model_patcher

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
    operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)
    state_dict = comfy.utils.load_torch_file(cfg["model_path"])
    with torch.device("meta"):
        predictor = apple_sharp.create_predictor(
            apple_sharp.PredictorParams(),
            dtype=dtype,
            device=None,
            operations=operations,
        )
    predictor.load_state_dict(state_dict, strict=False, assign=True)
    for name, buf in list(predictor.named_buffers()):
        if buf.device.type == "meta":
            parent = predictor
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")
    predictor.eval()
    if comfy.model_management.force_channels_last():
        predictor.to(memory_format=torch.channels_last)
    comfy.model_management.archive_model_dtypes(predictor)
    _model_patcher = comfy.model_patcher.ModelPatcher(
        predictor,
        load_device=load_device,
        offload_device=offload_device,
    )
    _model_config = cfg
    return _model_patcher


def image_to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    if image.dim() == 4:
        image = image[0]
    arr = image.detach().cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def focal_mm_to_px(width: int, height: int, focal_mm: float) -> float:
    focal = 30.0 if focal_mm <= 0 else float(focal_mm)
    diagonal_px = float((width * width + height * height) ** 0.5)
    return focal / 43.266615305567875 * diagonal_px


def _hash_image(image_np: np.ndarray) -> str:
    return hashlib.sha256(image_np.tobytes()).hexdigest()[:16]


def _monodepth_to(output: Any, device: Any):
    from .apple_sharp.model import MonodepthOutput
    return MonodepthOutput(
        disparity=output.disparity.to(device),
        encoder_features=[x.to(device) for x in output.encoder_features],
        decoder_features=output.decoder_features.to(device),
        output_features=[x.to(device) for x in output.output_features],
        intermediate_features=[x.to(device) for x in output.intermediate_features],
    )


@torch.no_grad()
def predict_gaussians(image_np: np.ndarray, focal_px: float, precision: str = "auto") -> Any:
    global _encode_cache
    import comfy.model_management

    from .apple_sharp import gaussians as gauss_mod
    patcher = load_model(precision)
    predictor = patcher.model
    device = patcher.load_device
    input_shape = [1, 3, INTERNAL_SIZE, INTERNAL_SIZE]
    comfy.model_management.load_models_gpu([patcher], memory_required=patcher.memory_required(input_shape))

    height, width = image_np.shape[:2]
    image_hash = _hash_image(image_np)
    if _encode_cache["image_hash"] == image_hash:
        monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
        image_resized = _encode_cache["image_resized"].to(device)
    else:
        image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
        image_resized = F.interpolate(
            image_pt[None],
            size=(INTERNAL_SIZE, INTERNAL_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        monodepth_output, _ = predictor.encode(image_resized)
        _encode_cache = {
            "image_hash": image_hash,
            "monodepth_output": _monodepth_to(monodepth_output, "cpu"),
            "image_resized": image_resized.detach().cpu(),
        }
        comfy.model_management.soft_empty_cache()

    disparity_factor = torch.tensor([float(focal_px) / float(width)], device=device)
    gaussians_ndc = predictor.decode(monodepth_output, image_resized, disparity_factor)
    intrinsics = torch.tensor(
        [
            [focal_px, 0, width / 2, 0],
            [0, focal_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    intrinsics[0] *= INTERNAL_SIZE / width
    intrinsics[1] *= INTERNAL_SIZE / height
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    return gauss_mod.unproject_gaussians(
        gaussians_ndc,
        extrinsics,
        intrinsics,
        (INTERNAL_SIZE, INTERNAL_SIZE),
    )


def filter_gaussians(gaussians: Any, max_gaussians: int, min_opacity: float) -> Any:
    if max_gaussians <= 0 and min_opacity <= 0:
        return gaussians
    opacity = gaussians.opacities.reshape(-1)
    mask = opacity >= float(min_opacity)
    if max_gaussians > 0 and int(mask.sum()) > max_gaussians:
        idx_all = torch.where(mask)[0]
        scores = opacity[idx_all]
        keep = torch.topk(scores, k=int(max_gaussians), largest=True).indices
        idx = idx_all[keep]
    else:
        idx = torch.where(mask)[0]
    return type(gaussians)(
        mean_vectors=gaussians.mean_vectors.reshape(-1, 3)[idx].unsqueeze(0),
        singular_values=gaussians.singular_values.reshape(-1, 3)[idx].unsqueeze(0),
        quaternions=gaussians.quaternions.reshape(-1, 4)[idx].unsqueeze(0),
        colors=gaussians.colors.reshape(-1, 3)[idx].unsqueeze(0),
        opacities=gaussians.opacities.reshape(-1)[idx].unsqueeze(0),
    )


def make_scene(
    image: torch.Tensor,
    *,
    precision: str,
    focal_length_mm: float,
    max_gaussians: int,
    min_opacity: float,
    save_ply: bool,
    output_prefix: str,
) -> SharpScene:
    from .apple_sharp import gaussians as gauss_mod
    image_np = image_to_numpy_rgb(image)
    height, width = image_np.shape[:2]
    focal_px = focal_mm_to_px(width, height, focal_length_mm)
    gaussians = predict_gaussians(image_np, focal_px, precision=precision)
    gaussians = filter_gaussians(gaussians, int(max_gaussians), float(min_opacity))
    pts = gaussians.mean_vectors.reshape(-1, 3).detach().float().cpu()
    bounds_min = torch.quantile(pts, 0.01, dim=0)
    bounds_max = torch.quantile(pts, 0.99, dim=0)
    center = (bounds_min + bounds_max) * 0.5
    radius = float(torch.linalg.norm(bounds_max - bounds_min).item() * 0.65)
    radius = max(radius, 0.5)
    ply_path = ""
    if save_ply:
        out_dir = Path(folder_paths.get_output_directory()) / "sharp_videoops"
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_path = str(out_dir / f"{output_prefix}_{int(time.time() * 1000)}.ply")
        gauss_mod.save_ply(gaussians, focal_px, (height, width), Path(ply_path))
    return SharpScene(
        gaussians=gaussians,
        focal_px=float(focal_px),
        source_size=(width, height),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        center=center,
        radius=radius,
        ply_path=ply_path,
    )


def look_at(position: torch.Tensor, target: torch.Tensor, roll: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    forward = target - position
    forward = forward / torch.linalg.norm(forward).clamp(min=1e-6)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    right = torch.cross(world_up, forward, dim=0)
    if torch.linalg.norm(right) < 1e-5:
        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    right = right / torch.linalg.norm(right).clamp(min=1e-6)
    up = torch.cross(forward, right, dim=0)
    if roll:
        c = float(np.cos(roll))
        s = float(np.sin(roll))
        right, up = right * c + up * s, -right * s + up * c
    return torch.stack([right, up, forward], dim=0), position


def camera_pose(scene: SharpScene, rig: dict[str, Any], t: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    preset = rig.get("preset", "cinematic_orbit")
    amp = float(rig.get("amplitude", 1.0))
    radius = scene.radius * float(rig.get("radius_scale", 1.35))
    yaw0 = np.deg2rad(float(rig.get("yaw", 0.0)))
    pitch0 = np.deg2rad(float(rig.get("pitch", 0.0)))
    phase = 2.0 * np.pi * (float(t) + float(rig.get("phase", 0.0)))
    target = scene.center.clone()
    roll = 0.0
    zoom = 1.0

    if preset == "dolly_push":
        radius *= 1.25 - 0.45 * t * amp
    elif preset == "dolly_pull":
        radius *= 0.8 + 0.55 * t * amp
    elif preset == "crane_up":
        target[1] += scene.radius * (t - 0.5) * 0.45 * amp
        pitch0 += np.deg2rad(10.0 * (t - 0.5) * amp)
    elif preset == "truck_left":
        target[0] += scene.radius * (t - 0.5) * 0.55 * amp
    elif preset == "arc_reveal":
        yaw0 += np.deg2rad((-18.0 + 36.0 * t) * amp)
        radius *= 1.1
    elif preset == "hero_parallax":
        yaw0 += np.deg2rad(np.sin(phase) * 8.0 * amp)
        pitch0 += np.deg2rad(np.cos(phase) * 3.0 * amp)
        zoom = 1.0 + 0.06 * np.sin(phase)
    elif preset == "micro_float":
        yaw0 += np.deg2rad(np.sin(phase) * 4.0 * amp)
        pitch0 += np.deg2rad(np.sin(phase * 0.7) * 2.0 * amp)
        target[1] += scene.radius * 0.04 * np.sin(phase * 1.3) * amp
    elif preset == "turntable":
        yaw0 += 2.0 * np.pi * t * amp
    elif preset == "custom":
        yaw0 += np.deg2rad(float(rig.get("yaw_delta", 0.0)) * t)
        pitch0 += np.deg2rad(float(rig.get("pitch_delta", 0.0)) * t)
        radius *= 1.0 + float(rig.get("radius_delta", 0.0)) * t
        roll = np.deg2rad(float(rig.get("roll_delta", 0.0)) * t)
    else:
        yaw0 += np.deg2rad(np.sin(phase) * 12.0 * amp)

    direction = torch.tensor(
        [
            np.sin(yaw0) * np.cos(pitch0),
            np.sin(pitch0),
            -np.cos(yaw0) * np.cos(pitch0),
        ],
        dtype=torch.float32,
    )
    position = target - direction * float(radius)
    rot, pos = look_at(position, target, roll=roll)
    return rot, pos, float(zoom)


def render_frame(
    scene: SharpScene,
    rig: dict[str, Any],
    t: float,
    width: int,
    height: int,
    *,
    splat_size: float,
    opacity_gain: float,
    exposure: float,
    gamma: float,
    background: tuple[float, float, float],
) -> torch.Tensor:
    g = scene.gaussians
    points = g.mean_vectors.reshape(-1, 3).detach().float().cpu()
    colors = g.colors.reshape(-1, 3).detach().float().cpu().clamp(0, 1)
    opacity = g.opacities.reshape(-1).detach().float().cpu().clamp(0, 1) * float(opacity_gain)
    scale = g.singular_values.reshape(-1, 3).detach().float().cpu().mean(dim=1)
    rot, pos, zoom = camera_pose(scene, rig, t)
    cam = (points - pos) @ rot.T
    z = cam[:, 2]
    valid = z > 1e-4
    if not valid.any():
        bg = torch.tensor(background, dtype=torch.float32)
        return bg.reshape(1, 1, 3).repeat(height, width, 1)
    cam = cam[valid]
    z = z[valid]
    colors = colors[valid]
    opacity = opacity[valid]
    scale = scale[valid]
    focal = float(scene.focal_px) * min(width / scene.source_size[0], height / scene.source_size[1]) * zoom
    x = cam[:, 0] / z * focal + width * 0.5
    y = -cam[:, 1] / z * focal + height * 0.5
    radius = (scale / z.abs().clamp(min=1e-4) * focal * splat_size).clamp(0.6, 6.0)
    xi = x.round().long()
    yi = y.round().long()
    in_view = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    if not in_view.any():
        bg = torch.tensor(background, dtype=torch.float32)
        return bg.reshape(1, 1, 3).repeat(height, width, 1)
    xi, yi, z = xi[in_view], yi[in_view], z[in_view]
    colors, opacity, radius = colors[in_view], opacity[in_view], radius[in_view]
    order = torch.argsort(z, descending=True)
    canvas = torch.tensor(background, dtype=torch.float32).reshape(1, 1, 3).repeat(height, width, 1)
    alpha_canvas = torch.zeros(height, width, dtype=torch.float32)
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    for idx in order.tolist():
        r = int(max(1, min(3, round(float(radius[idx])))))
        for ox, oy in offsets[: 1 + min(8, (r - 1) * 4)]:
            px = int(xi[idx]) + ox
            py = int(yi[idx]) + oy
            if 0 <= px < width and 0 <= py < height:
                dist = (ox * ox + oy * oy) ** 0.5 / max(float(r), 1.0)
                a = float(opacity[idx]) * float(np.exp(-dist * dist * 1.8))
                a = max(0.0, min(1.0, a))
                cur_a = alpha_canvas[py, px]
                out_a = a + cur_a * (1.0 - a)
                if out_a > 1e-6:
                    canvas[py, px] = (colors[idx] * a + canvas[py, px] * cur_a * (1.0 - a)) / out_a
                    alpha_canvas[py, px] = out_a
    canvas = (canvas * float(exposure)).clamp(0, 1)
    if gamma > 0:
        canvas = canvas.pow(1.0 / float(gamma))
    return canvas.clamp(0, 1)
