"""
DepthFlow integration helpers for ComfyUI nodes.

This module keeps DepthFlow-specific path discovery, estimator invocation,
depth-map preparation, and state mapping in one place so ShaderFlow nodes do
not need to duplicate renderer details.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


logger = logging.getLogger("DepthFlow.Adapter")


def _node_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _own_root() -> Path:
    return _node_root().parents[2]


def setup_depthflow_paths() -> list[Path]:
    """Add local DepthFlow install/source paths to sys.path."""
    candidates = [
        _node_root() / ".venv_depthflow",
        _own_root() / "DepthFlow",
        _own_root() / "ShaderFlow",
        Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/DepthFlow"),
        Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/ShaderFlow"),
    ]
    added: list[Path] = []
    for base in candidates:
        if not base.exists():
            continue
        site_dirs = []
        if base.name.startswith(".venv"):
            site_dirs.extend(base.glob("lib/python*/site-packages"))
            site_dirs.extend(base.glob("Lib/site-packages"))
        for p in [base, *site_dirs]:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                added.append(p)
    return added


def find_cuda_renderer() -> Any | None:
    """Return the local depthflow.cuda_renderer module when CUDA is usable."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    setup_depthflow_paths()
    search_roots = [
        _node_root() / ".venv_depthflow",
        _own_root() / "DepthFlow",
        Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/DepthFlow"),
    ]
    for base in search_roots:
        if not base.exists():
            continue
        candidates = [base / "depthflow" / "cuda_renderer.py"]
        candidates.extend((base / "lib").glob("python*/site-packages/depthflow/cuda_renderer.py"))
        candidates.extend((base / "Lib" / "site-packages").glob("depthflow/cuda_renderer.py"))
        for candidate in candidates:
            if candidate.is_file():
                spec = importlib.util.spec_from_file_location("depthflow.cuda_renderer", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    try:
        from depthflow import cuda_renderer
        return cuda_renderer
    except ImportError:
        return None


def depth_image_to_numpy(depth_map: Any) -> np.ndarray:
    """Convert a ComfyUI IMAGE depth tensor/array to a 2D float array."""
    if HAS_TORCH and isinstance(depth_map, torch.Tensor):
        data = depth_map.detach().cpu().numpy()
    else:
        data = np.asarray(depth_map)
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        data = data[:, :, 0]
    return np.asarray(data, dtype=np.float32)


def _normalise_01(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.all():
        depth = np.where(finite, depth, 0.0).astype(np.float32)
    dmin = float(depth.min())
    dmax = float(depth.max())
    if dmax <= dmin:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - dmin) / (dmax - dmin)).astype(np.float32)


def prepare_depth_map(
    depth_np: np.ndarray,
    src_img: np.ndarray | None = None,
    *,
    normalize_mode: str = "auto",
    invert_depth: float = 0.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Prepare a depth map for DepthFlow while preserving user intent.

    normalize_mode:
      - none: only cast and clip
      - auto: scale 8/16-bit style maps, preserve already-normalized floats
      - minmax: always normalize by current min/max
    """
    depth = np.asarray(depth_np, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    mode = (normalize_mode or "auto").lower()
    if mode == "minmax":
        depth = _normalise_01(depth)
    elif mode == "auto":
        finite_max = float(np.nanmax(depth)) if depth.size else 0.0
        finite_min = float(np.nanmin(depth)) if depth.size else 0.0
        if finite_max > 255.0:
            depth = depth / 65535.0
        elif finite_max > 1.5 or finite_min < -0.001:
            depth = depth / 255.0
    elif mode != "none":
        raise ValueError(f"Unknown DepthFlow normalize mode: {normalize_mode}")

    depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    depth = np.clip(depth, 0.0, 1.0)

    inv = float(invert_depth)
    if inv:
        inv = max(0.0, min(1.0, inv))
        depth = depth * (1.0 - inv) + (1.0 - depth) * inv

    if smooth_sigma and smooth_sigma > 0:
        try:
            import cv2
            if src_img is not None and hasattr(cv2, "ximgproc"):
                guide = np.asarray(src_img, dtype=np.uint8)
                depth = cv2.ximgproc.guidedFilter(
                    guide, depth, radius=max(2, int(smooth_sigma * 4)), eps=1e-3,
                )
            else:
                k = max(3, int(smooth_sigma * 6) | 1)
                depth = cv2.GaussianBlur(
                    depth, (k, k), sigmaX=smooth_sigma, sigmaY=smooth_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
        except Exception as exc:
            logger.info("Depth smoothing skipped: %s", exc)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def estimate_depth(
    img_np: np.ndarray,
    estimator: str = "da2",
    *,
    model_size: str = "small",
    da3_resolution: int = 1024,
    postprocess: bool = True,
    sigma: float | None = None,
    thicken: int | None = None,
    allow_luminance_fallback: bool = False,
) -> np.ndarray:
    """Estimate depth through the local DepthFlow estimator classes."""
    setup_depthflow_paths()
    key = (estimator or "da2").lower()
    try:
        if key in ("da1", "da2", "da3"):
            from depthflow.estimators.anything import (
                DepthAnythingV1,
                DepthAnythingV2,
                DepthAnythingV3,
            )
            cls = {"da1": DepthAnythingV1, "da2": DepthAnythingV2, "da3": DepthAnythingV3}[key]
            kwargs: dict[str, Any] = {"post": bool(postprocess)}
            if hasattr(cls, "Model"):
                safe_size = "large" if key != "da3" and model_size == "giant" else model_size
                kwargs["model"] = cls.Model(safe_size)
            if key == "da3":
                kwargs["resolution"] = int(da3_resolution)
            if sigma is not None:
                kwargs["sigma"] = float(sigma)
            if thicken is not None:
                kwargs["thicken"] = int(thicken)
            depth = cls(**kwargs).estimate(img_np)
        elif key == "depthpro":
            from depthflow.estimators.depthpro import DepthPro
            depth = DepthPro(post=bool(postprocess)).estimate(img_np)
        elif key == "zoedepth":
            from depthflow.estimators.zoedepth import ZoeDepth
            depth = ZoeDepth(post=bool(postprocess)).estimate(img_np)
        elif key == "marigold":
            from depthflow.estimators.marigold import Marigold
            depth = Marigold(post=bool(postprocess)).estimate(img_np)
        else:
            raise ValueError(f"Unknown DepthFlow estimator: {estimator}")
        return prepare_depth_map(depth, img_np, normalize_mode="auto")
    except Exception:
        if not allow_luminance_fallback:
            raise
        gray = np.mean(img_np.astype(np.float32), axis=2)
        return 1.0 - (gray / max(float(gray.max()), 1.0))


def set_depth_pair(state: Any, attr: str, x: float, y: float, *, add: bool = False) -> None:
    """Set or add tuple/vector fields on native DepthState or CUDA state."""
    x = float(x)
    y = float(y)
    if hasattr(state, attr):
        try:
            if add:
                cur = getattr(state, attr)
                setattr(state, attr, (float(cur[0]) + x, float(cur[1]) + y))
            else:
                setattr(state, attr, (x, y))
            return
        except Exception:
            pass
    if add:
        x += float(getattr(state, f"{attr}_x", 0.0))
        y += float(getattr(state, f"{attr}_y", 0.0))
    setattr(state, f"{attr}_x", x)
    setattr(state, f"{attr}_y", y)


def apply_depth_state(state: Any, params: dict[str, Any]) -> None:
    """Apply DepthFlow controls consistently to native and CUDA states."""
    height_mode = params.get("height_mode", "motion_preset")
    if height_mode == "override":
        state.height = float(params.get("depth_height", state.height))
    elif height_mode == "multiply":
        state.height *= float(params.get("depth_height", 1.0))

    state.steady = float(params.get("steady_depth", getattr(state, "steady", 0.15)))
    state.focus = float(params.get("focus_depth", getattr(state, "focus", 0.0)))
    state.zoom = float(params.get("zoom", getattr(state, "zoom", 1.0)))
    state.isometric = float(params.get("isometric", getattr(state, "isometric", 0.0)))
    state.dolly = float(params.get("dolly", getattr(state, "dolly", 0.0)))
    if hasattr(state, "mirror"):
        state.mirror = bool(params.get("mirror", state.mirror))

    set_depth_pair(state, "offset", params.get("offset_x", 0.0), params.get("offset_y", 0.0), add=True)
    set_depth_pair(state, "center", params.get("center_x", 0.0), params.get("center_y", 0.0))
    set_depth_pair(state, "origin", params.get("origin_x", 0.0), params.get("origin_y", 0.0))


def backend_order(policy: str) -> tuple[str, ...]:
    """Resolve explicit backend policy without silent low-quality fallback."""
    return {
        "cuda_first": ("cuda", "opengl"),
        "opengl_first": ("opengl", "cuda"),
        "cuda_only": ("cuda",),
        "opengl_only": ("opengl",),
        "auto": ("cuda", "opengl"),
        "allow_cpu": ("cuda", "opengl", "cpu"),
    }.get(policy, ("cuda", "opengl"))


def high_quality_backend_error(policy: str, errors: list[str]) -> RuntimeError:
    detail = "\n".join(f"  - {item}" for item in errors) or "  - no backend attempted"
    return RuntimeError(
        "DepthFlow high-quality backend unavailable. "
        f"backend_policy={policy}. CUDA/OpenGL failed:\n{detail}\n"
        "Set backend_policy=allow_cpu only if a low-quality fallback is acceptable."
    )
