"""
DepthFlow integration helpers for ComfyUI nodes.

This module keeps DepthFlow-specific path discovery, estimator invocation,
depth-map preparation, and state mapping in one place so ShaderFlow nodes do
not need to duplicate renderer details.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
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


def _venv_python() -> Path | None:
    venv = _node_root() / ".venv_depthflow"
    candidates = [
        venv / "bin" / "python",
        venv / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _isolate_depthflow_import_path() -> list[Path]:
    """Prioritize the DepthFlow venv and avoid stale attr/attrs from host Python."""
    venv = _node_root() / ".venv_depthflow"
    site_dirs = []
    if venv.is_dir():
        site_dirs.extend(venv.glob("lib/python*/site-packages"))
        site_dirs.extend(venv.glob("Lib/site-packages"))

    added: list[Path] = []
    for site_dir in reversed([p for p in site_dirs if p.is_dir()]):
        site_str = str(site_dir)
        sys.path[:] = [p for p in sys.path if p != site_str]
        sys.path.insert(0, site_str)
        added.append(site_dir)

    sys.path[:] = [
        p for p in sys.path
        if not (
            ("python3.10" in p or "Python310" in p or "python310" in p)
            and ("site-packages" in p or "dist-packages" in p)
            and str(_node_root()) not in p
        )
    ]

    for name in ("attr", "attrs"):
        mod = sys.modules.get(name)
        mod_file = str(getattr(mod, "__file__", "") or "")
        if mod is not None and not any(str(site) in mod_file for site in site_dirs):
            del sys.modules[name]
    return added


def setup_depthflow_paths() -> list[Path]:
    """Add local DepthFlow install/source paths to sys.path."""
    added = _isolate_depthflow_import_path()
    candidates = [
        _node_root() / ".venv_depthflow",
        _own_root() / "DepthFlow",
        _own_root() / "ShaderFlow",
        Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/DepthFlow"),
        Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/ShaderFlow"),
    ]
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


def _estimate_depth_subprocess(
    img_np: np.ndarray,
    estimator: str,
    *,
    model_size: str,
    da3_resolution: int,
    postprocess: bool,
    sigma: float | None,
    thicken: int | None,
) -> np.ndarray | None:
    py = _venv_python()
    if py is None:
        return None

    worker = Path(__file__).resolve().parent / "depthflow_estimate_worker.py"
    if not worker.is_file():
        return None

    with tempfile.TemporaryDirectory(prefix="vhs_depthflow_est_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.npy"
        output_path = tmp_dir / "depth.npy"
        params_path = tmp_dir / "params.json"
        np.save(input_path, np.asarray(img_np, dtype=np.uint8))
        params_path.write_text(json.dumps({
            "estimator": estimator,
            "model_size": model_size,
            "da3_resolution": int(da3_resolution),
            "postprocess": bool(postprocess),
            "sigma": sigma,
            "thicken": thicken,
        }), encoding="utf-8")

        cmd = [str(py), str(worker), str(input_path), str(output_path), str(params_path)]

        def run_worker(label: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
            if output_path.exists():
                output_path.unlink()
            env = os.environ.copy()
            env.setdefault("PYOPENGL_PLATFORM", "egl")
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            env["VHS_DEPTHFLOW_ESTIMATE_DEVICE"] = label
            if env_overrides:
                env.update(env_overrides)
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        device_mode = os.environ.get("VHS_DEPTHFLOW_ESTIMATE_DEVICE", "auto").strip().lower()
        cpu_env = {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        }

        if device_mode == "cpu":
            result = run_worker("cpu", cpu_env)
        else:
            result = run_worker("cuda")
            if result.returncode != 0 and device_mode in {"auto", ""}:
                detail_probe = "\n".join(x for x in [result.stdout.strip(), result.stderr.strip()] if x)
                cuda_busy = any(
                    needle in detail_probe
                    for needle in (
                        "CUDA-capable device(s) is/are busy or unavailable",
                        "CUDA out of memory",
                        "CUDA error",
                        "cudaGetDeviceCount",
                    )
                )
                if cuda_busy:
                    logger.info("DepthFlow CUDA estimator failed; retrying depth estimation on CPU")
                    result = run_worker("cpu", cpu_env)

        if result.returncode != 0:
            detail = "\n".join(x for x in [result.stdout.strip(), result.stderr.strip()] if x)
            raise RuntimeError(
                "DepthFlow subprocess depth estimation failed. "
                f"python={py}\n{detail[-4000:]}"
            )
        return np.load(output_path).astype(np.float32)


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
    mode = os.environ.get("VHS_DEPTHFLOW_ESTIMATE_MODE", "subprocess").strip().lower()
    if mode in {"subprocess", "auto", ""}:
        depth = _estimate_depth_subprocess(
            img_np,
            estimator,
            model_size=model_size,
            da3_resolution=da3_resolution,
            postprocess=postprocess,
            sigma=sigma,
            thicken=thicken,
        )
        if depth is not None:
            return prepare_depth_map(depth, img_np, normalize_mode="auto")
        if mode == "subprocess":
            logger.info("DepthFlow subprocess venv not found; falling back to in-process estimator")

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
