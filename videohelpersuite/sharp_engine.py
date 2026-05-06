"""Independent Apple SHARP engine for VideoHelperSuite video nodes."""
from __future__ import annotations

import hashlib
import logging
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


logger = logging.getLogger("VHS.SHARP")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
_gsplat_warning_emitted = False
_gsplat_info_emitted = False


@dataclass
class SharpScene:
    gaussians: Any
    focal_px: float
    source_size: tuple[int, int]
    source_image: torch.Tensor
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


def log_info(message: str) -> None:
    logger.info(message)


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
    log_info(f"Model config: repo={SHARP_REPO_ID}, file={SHARP_FILENAME}, path={path}, dtype={dtype_str}")
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
        log_info("Model cache hit: reusing loaded RGBGaussianPredictor")
        return _model_patcher

    t0 = time.perf_counter()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
    operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)
    state_dict = comfy.utils.load_torch_file(cfg["model_path"])
    log_info(f"Loading RGBGaussianPredictor: precision={precision}, dtype={cfg['dtype']}, device={load_device}")
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
    log_info(f"Model initialized in {time.perf_counter() - t0:.2f}s")
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
    log_info(f"SHARP inference input: source={width}x{height}, internal={INTERNAL_SIZE}x{INTERNAL_SIZE}, focal_px={focal_px:.2f}")
    image_hash = _hash_image(image_np)
    if _encode_cache["image_hash"] == image_hash:
        log_info("Encode cache hit: reusing monodepth features")
        monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
        image_resized = _encode_cache["image_resized"].to(device)
    else:
        t_encode = time.perf_counter()
        log_info("Encode start: resizing image and running SHARP encoder")
        image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
        image_resized = F.interpolate(
            image_pt[None],
            size=(INTERNAL_SIZE, INTERNAL_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        monodepth_output, _ = predictor.encode(image_resized)
        log_info(f"Encode complete in {time.perf_counter() - t_encode:.2f}s")
        _encode_cache = {
            "image_hash": image_hash,
            "monodepth_output": _monodepth_to(monodepth_output, "cpu"),
            "image_resized": image_resized.detach().cpu(),
        }
        comfy.model_management.soft_empty_cache()

    disparity_factor = torch.tensor([float(focal_px) / float(width)], device=device)
    t_decode = time.perf_counter()
    log_info("Decode start: generating 3D Gaussians")
    gaussians_ndc = predictor.decode(monodepth_output, image_resized, disparity_factor)
    log_info(f"Decode complete in {time.perf_counter() - t_decode:.2f}s")
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
    gaussians = gauss_mod.unproject_gaussians(
        gaussians_ndc,
        extrinsics,
        intrinsics,
        (INTERNAL_SIZE, INTERNAL_SIZE),
    )
    log_info(f"Unproject complete: gaussians={gaussians.mean_vectors.reshape(-1, 3).shape[0]}")
    return gaussians


def filter_gaussians(gaussians: Any, max_gaussians: int, min_opacity: float) -> Any:
    if max_gaussians <= 0 and min_opacity <= 0:
        return gaussians
    before = int(gaussians.opacities.reshape(-1).shape[0])
    opacity = gaussians.opacities.reshape(-1)
    mask = opacity >= float(min_opacity)
    if max_gaussians > 0 and int(mask.sum()) > max_gaussians:
        idx_all = torch.where(mask)[0]
        scores = opacity[idx_all]
        keep = torch.topk(scores, k=int(max_gaussians), largest=True).indices
        idx = idx_all[keep]
    else:
        idx = torch.where(mask)[0]
    filtered = type(gaussians)(
        mean_vectors=gaussians.mean_vectors.reshape(-1, 3)[idx].unsqueeze(0),
        singular_values=gaussians.singular_values.reshape(-1, 3)[idx].unsqueeze(0),
        quaternions=gaussians.quaternions.reshape(-1, 4)[idx].unsqueeze(0),
        colors=gaussians.colors.reshape(-1, 3)[idx].unsqueeze(0),
        opacities=gaussians.opacities.reshape(-1)[idx].unsqueeze(0),
    )
    log_info(f"Gaussian filter: before={before}, after={int(idx.shape[0])}, max={max_gaussians}, min_opacity={min_opacity}")
    return filtered


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
    log_info(
        f"Build scene start: source={width}x{height}, precision={precision}, "
        f"focal_mm={focal_length_mm}, max_gaussians={max_gaussians}, min_opacity={min_opacity}"
    )
    t0 = time.perf_counter()
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
        log_info(f"PLY saved: {ply_path}")
    log_info(f"Build scene complete in {time.perf_counter() - t0:.2f}s, radius={radius:.4f}")
    return SharpScene(
        gaussians=gaussians,
        focal_px=float(focal_px),
        source_size=(width, height),
        source_image=torch.from_numpy(image_np).float() / 255.0,
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


def render_device(name: str = "auto") -> torch.device:
    mode = str(name or "auto").lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(str(device))
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def splat_offsets(mode: str, device: torch.device) -> torch.Tensor:
    quality = str(mode or "balanced").lower()
    if quality == "point":
        offsets = [(0, 0)]
    elif quality == "fast":
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
        ]
    return torch.tensor(offsets, dtype=torch.float32, device=device)


def _background_name(background: tuple[float, float, float]) -> str:
    if sum(background) / 3.0 > 0.5:
        return "white"
    return "black"


def _sharp_intrinsics(scene: SharpScene, width: int, height: int, device: torch.device) -> torch.Tensor:
    src_w, src_h = scene.source_size
    fx = float(scene.focal_px) * float(width) / max(float(src_w), 1.0)
    fy = float(scene.focal_px) * float(height) / max(float(src_h), 1.0)
    intrinsics = torch.eye(4, dtype=torch.float32, device=device)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = float(width) * 0.5
    intrinsics[1, 2] = float(height) * 0.5
    return intrinsics


def _depth_focus(scene: SharpScene, device: torch.device) -> float:
    points = scene.gaussians.mean_vectors.reshape(-1, 3).detach().float()
    z = points[:, 2]
    z = z[z > 1e-4]
    if z.numel() == 0:
        return 2.0
    return max(2.0, float(torch.quantile(z.cpu(), 0.1)))


def _max_eye_offset(scene: SharpScene, width: int, height: int, amplitude: float, radius_scale: float) -> np.ndarray:
    points = scene.gaussians.mean_vectors.reshape(-1, 3).detach().float()
    z = points[:, 2]
    z = z[z > 1e-4]
    min_depth = float(torch.quantile(z.cpu(), 0.001)) if z.numel() else 2.0
    diagonal = np.sqrt((float(width) / float(scene.focal_px)) ** 2 + (float(height) / float(scene.focal_px)) ** 2)
    lateral = 0.08 * diagonal * min_depth * max(0.0, float(amplitude)) * max(0.1, float(radius_scale))
    medial = 0.15 * min_depth * max(0.0, float(amplitude))
    return np.array([lateral, lateral, medial], dtype=np.float32)


def _camera_matrix(
    position: torch.Tensor,
    look_at_position: torch.Tensor,
    world_up: torch.Tensor,
    inverse: bool,
) -> torch.Tensor:
    camera_front = look_at_position - position
    camera_front = camera_front / camera_front.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    camera_right = torch.cross(camera_front, world_up, dim=-1)
    camera_right = camera_right / camera_right.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    camera_down = torch.cross(camera_front, camera_right, dim=-1)
    rotation = torch.stack([camera_right, camera_down, camera_front], dim=-1)
    matrix = torch.eye(4, dtype=torch.float32, device=position.device)
    if inverse:
        matrix[:3, :3] = rotation.T
        matrix[:3, 3:4] = -rotation.T @ position[:, None]
    else:
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position
    return matrix


def _official_eye_position(scene: SharpScene, rig: dict[str, Any], t: float, width: int, height: int, device: torch.device) -> torch.Tensor:
    preset = str(rig.get("preset", "cinematic_orbit"))
    amp = float(rig.get("amplitude", 1.0))
    radius_scale = float(rig.get("radius_scale", 1.0))
    phase = 2.0 * np.pi * (float(t) + float(rig.get("phase", 0.0)))
    offset_x, offset_y, offset_z = _max_eye_offset(scene, width, height, amp, radius_scale)
    yaw_delta = np.deg2rad(float(rig.get("yaw_delta", 20.0)) * float(t))
    pitch_delta = np.deg2rad(float(rig.get("pitch_delta", 0.0)) * float(t))
    radius_delta = float(rig.get("radius_delta", 0.0)) * float(t)

    if preset in {"cinematic_orbit", "hero_parallax"}:
        eye = [offset_x * np.sin(phase), 0.0, offset_z * (1.0 - np.cos(phase)) * 0.5]
    elif preset == "micro_float":
        eye = [offset_x * 0.45 * np.sin(phase), offset_y * 0.35 * np.sin(phase * 0.7), offset_z * 0.25 * (1.0 - np.cos(phase))]
    elif preset == "turntable":
        eye = [offset_x * np.sin(phase), offset_y * np.cos(phase), 0.0]
    elif preset == "truck_left":
        eye = [offset_x * (2.0 * t - 1.0), 0.0, 0.0]
    elif preset == "dolly_push":
        eye = [0.0, 0.0, -offset_z * float(t)]
    elif preset == "dolly_pull":
        eye = [0.0, 0.0, offset_z * float(t)]
    elif preset == "crane_up":
        eye = [0.0, offset_y * (float(t) - 0.5), 0.0]
    elif preset == "arc_reveal":
        local_phase = np.deg2rad(-55.0 + 110.0 * float(t))
        eye = [offset_x * np.sin(local_phase), 0.0, offset_z * (1.0 - np.cos(local_phase)) * 0.5]
    elif preset == "custom":
        eye = [
            offset_x * np.sin(yaw_delta),
            offset_y * np.sin(pitch_delta),
            offset_z * radius_delta,
        ]
    else:
        eye = [offset_x * np.sin(phase), 0.0, offset_z * (1.0 - np.cos(phase)) * 0.5]
    return torch.tensor(eye, dtype=torch.float32, device=device)


def _official_extrinsics(scene: SharpScene, rig: dict[str, Any], t: float, width: int, height: int, device: torch.device) -> torch.Tensor:
    eye = _official_eye_position(scene, rig, t, width, height, device)
    mode = "ahead" if str(rig.get("lookat_mode", "point")) == "ahead" else "point"
    origin = eye if mode == "ahead" else torch.zeros(3, dtype=torch.float32, device=device)
    focus = _depth_focus(scene, device)
    look_at = origin + torch.tensor([0.0, 0.0, focus], dtype=torch.float32, device=device)
    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
    return _camera_matrix(eye, look_at, world_up, inverse=True)


def _render_frame_gsplat(
    scene: SharpScene,
    rig: dict[str, Any],
    t: float,
    width: int,
    height: int,
    background: tuple[float, float, float],
    render_backend: str,
    render_mode: str,
) -> torch.Tensor | None:
    global _gsplat_info_emitted, _gsplat_warning_emitted
    device = render_device(render_backend)
    if device.type != "cuda":
        if not _gsplat_warning_emitted:
            log_info(f"Official gsplat renderer skipped: requires CUDA, current device={device}")
            _gsplat_warning_emitted = True
        return None
    try:
        import gsplat
        from .apple_sharp import color_space as cs_utils
    except Exception as exc:
        if not _gsplat_warning_emitted:
            log_info(f"Official gsplat renderer unavailable: {exc}. Falling back to torch preview renderer.")
            _gsplat_warning_emitted = True
        return None

    if not _gsplat_info_emitted:
        log_info(
            f"Official gsplat renderer active: device={device}, size={width}x{height}, "
            f"mode={render_mode}, camera_preset={rig.get('preset', 'custom')}"
        )
        _gsplat_info_emitted = True

    g = scene.gaussians.to(device)
    intrinsics = _sharp_intrinsics(scene, width, height, device)
    extrinsics = _official_extrinsics(scene, rig, t, width, height, device)
    colors, alphas, _ = gsplat.rendering.rasterization(
        means=g.mean_vectors[0],
        quats=g.quaternions[0],
        scales=g.singular_values[0],
        opacities=g.opacities[0],
        colors=g.colors[0],
        viewmats=extrinsics[None],
        Ks=intrinsics[None, :3, :3],
        width=int(width),
        height=int(height),
        render_mode="RGB+D",
        rasterize_mode="classic",
        absgrad=False,
        packed=False,
        eps2d=0.3,
    )
    rendered_color = colors[..., 0:3].permute(0, 3, 1, 2)
    rendered_depth_unnormalized = colors[..., 3:4].permute(0, 3, 1, 2)
    rendered_alpha = alphas.permute(0, 3, 1, 2)
    mode = str(render_mode or "photo_composite").lower()
    if mode == "alpha":
        out = rendered_alpha.expand(-1, 3, -1, -1)
    elif mode == "depth":
        depth = rendered_depth_unnormalized / torch.clip(rendered_alpha, min=1e-8)
        valid = rendered_alpha > 1e-4
        if valid.any():
            d = depth[valid]
            dmin, dmax = torch.quantile(d, 0.02), torch.quantile(d, 0.98)
            depth = ((depth - dmin) / (dmax - dmin).clamp(min=1e-6)).clamp(0, 1)
        out = depth.expand(-1, 3, -1, -1)
    else:
        bg_name = _background_name(background)
        if bg_name == "white":
            rendered_color = rendered_color + (1.0 - rendered_alpha)
        rendered_color = cs_utils.linearRGB2sRGB(rendered_color.clamp(0, 1))
        out = rendered_color
    return out[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()


def _render_photo_warp_fallback(
    scene: SharpScene,
    rig: dict[str, Any],
    t: float,
    width: int,
    height: int,
    device: torch.device,
    exposure: float,
    gamma: float,
) -> torch.Tensor:
    src = scene.source_image.to(device).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(src, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(height), dtype=torch.float32, device=device),
        torch.linspace(-1.0, 1.0, int(width), dtype=torch.float32, device=device),
        indexing="ij",
    )
    eye = _official_eye_position(scene, rig, t, width, height, device)
    focus = max(_depth_focus(scene, device), 1e-3)
    parallax_x = (eye[0] / focus).clamp(-0.2, 0.2)
    parallax_y = (eye[1] / focus).clamp(-0.2, 0.2)
    radial = (xx.square() + yy.square()).sqrt().clamp(0, 1)
    depth_proxy = 0.25 + 0.75 * radial
    grid = torch.stack(
        [
            xx + parallax_x * depth_proxy * 1.6,
            yy - parallax_y * depth_proxy * 1.6,
        ],
        dim=-1,
    ).unsqueeze(0)
    warped = F.grid_sample(resized, grid, mode="bilinear", padding_mode="border", align_corners=True)
    canvas = warped[0].permute(1, 2, 0).clamp(0, 1)
    canvas = (canvas * float(exposure)).clamp(0, 1)
    return canvas.detach().cpu()


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
    render_backend: str = "auto",
    splat_quality: str = "balanced",
    render_mode: str = "photo_composite",
    source_photo_strength: float = 0.85,
) -> torch.Tensor:
    gsplat_frame = _render_frame_gsplat(scene, rig, t, width, height, background, render_backend, render_mode)
    if gsplat_frame is not None:
        canvas = (gsplat_frame * float(exposure)).clamp(0, 1)
        return canvas.clamp(0, 1)

    device = render_device(render_backend)
    if str(render_mode or "photo_composite").lower() == "photo_composite":
        return _render_photo_warp_fallback(scene, rig, t, width, height, device, exposure, gamma)

    g = scene.gaussians
    points = g.mean_vectors.reshape(-1, 3).detach().float().to(device)
    colors = g.colors.reshape(-1, 3).detach().float().to(device).clamp(0, 1)
    opacity = g.opacities.reshape(-1).detach().float().to(device).clamp(0, 1) * float(opacity_gain)
    scale = g.singular_values.reshape(-1, 3).detach().float().to(device).mean(dim=1)
    rot, pos, zoom = camera_pose(scene, rig, t)
    rot = rot.to(device)
    pos = pos.to(device)
    cam = (points - pos) @ rot.T
    z = cam[:, 2]
    valid = z > 1e-4
    if not valid.any():
        return render_background(scene, width, height, background, device, render_mode, source_photo_strength).cpu()
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
        return render_background(scene, width, height, background, device, render_mode, source_photo_strength).cpu()
    xi, yi, z = xi[in_view], yi[in_view], z[in_view]
    colors, opacity, radius = colors[in_view], opacity[in_view], radius[in_view].clamp(0.75, 6.0)

    offsets = splat_offsets(splat_quality, device)
    ox = offsets[:, 0].reshape(1, -1)
    oy = offsets[:, 1].reshape(1, -1)
    px = xi.reshape(-1, 1) + ox.long()
    py = yi.reshape(-1, 1) + oy.long()
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    dist = torch.sqrt(ox.square() + oy.square()) / radius.reshape(-1, 1).clamp(min=1.0)
    weight = opacity.reshape(-1, 1) * torch.exp(-dist.square() * 1.8)
    near_boost = (1.0 / z.reshape(-1, 1).clamp(min=0.05)).clamp(max=4.0)
    weight = (weight * near_boost).clamp(0.0, 1.0)

    flat_idx = (py * int(width) + px).reshape(-1)
    flat_inside = inside.reshape(-1)
    flat_idx = flat_idx[flat_inside].long()
    flat_weight = weight.reshape(-1)[flat_inside]
    flat_colors = colors[:, None, :].expand(-1, offsets.shape[0], -1).reshape(-1, 3)[flat_inside]

    num_pixels = int(width) * int(height)
    alpha_accum = torch.zeros(num_pixels, dtype=torch.float32, device=device)
    color_accum = torch.zeros(num_pixels, 3, dtype=torch.float32, device=device)
    alpha_accum.scatter_add_(0, flat_idx, flat_weight)
    color_accum.scatter_add_(0, flat_idx[:, None].expand(-1, 3), flat_colors * flat_weight[:, None])

    bg_image = render_background(scene, width, height, background, device, render_mode, source_photo_strength).reshape(-1, 3)
    alpha = alpha_accum.clamp(0.0, 1.0).reshape(-1, 1)
    splat_color = color_accum / alpha_accum.clamp(min=1e-6).reshape(-1, 1)
    mode = str(render_mode or "photo_composite").lower()
    if mode == "alpha":
        canvas = alpha.expand(-1, 3)
    elif mode == "depth":
        depth = torch.zeros(int(width) * int(height), dtype=torch.float32, device=device)
        inv_z = (1.0 / z.clamp(min=0.05)).reshape(-1, 1).expand(-1, offsets.shape[0]).reshape(-1)[flat_inside]
        depth.scatter_add_(0, flat_idx, inv_z * flat_weight)
        depth = depth / alpha_accum.clamp(min=1e-6)
        valid_depth = alpha_accum > 1e-6
        if valid_depth.any():
            d = depth[valid_depth]
            dmin, dmax = torch.quantile(d, 0.02), torch.quantile(d, 0.98)
            depth = ((depth - dmin) / (dmax - dmin).clamp(min=1e-6)).clamp(0, 1)
        canvas = depth.reshape(-1, 1).expand(-1, 3)
    elif mode == "gaussian_color":
        flat_bg = torch.tensor(background, dtype=torch.float32, device=device).reshape(1, 3)
        canvas = splat_color * alpha + flat_bg * (1.0 - alpha)
    else:
        strength = max(0.0, min(1.0, float(source_photo_strength)))
        composite_alpha = (alpha * (1.0 - strength * 0.35)).clamp(0.0, 1.0)
        canvas = splat_color * composite_alpha + bg_image * (1.0 - composite_alpha)
    canvas = canvas.reshape(int(height), int(width), 3)
    canvas = (canvas * float(exposure)).clamp(0, 1)
    if gamma > 0:
        canvas = canvas.pow(1.0 / float(gamma))
    return canvas.clamp(0, 1).cpu()


def render_background(
    scene: SharpScene,
    width: int,
    height: int,
    background: tuple[float, float, float],
    device: torch.device,
    render_mode: str,
    source_photo_strength: float,
) -> torch.Tensor:
    mode = str(render_mode or "photo_composite").lower()
    if mode == "photo_composite" and source_photo_strength > 0:
        src = scene.source_image.to(device).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(src, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        return resized[0].permute(1, 2, 0).clamp(0, 1)
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    return bg.reshape(1, 1, 3).repeat(int(height), int(width), 1)
