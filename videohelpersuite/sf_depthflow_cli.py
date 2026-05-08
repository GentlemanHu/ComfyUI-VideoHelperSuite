"""ShaderFlow DepthFlow CLI bridge.

This keeps the high-quality OpenGL path in the same isolated DepthFlow venv
used by VHS_DepthFlow_Generator. ComfyUI only reads frames from the rendered
video, so it never imports DepthFlow/ShaderFlow packages in the main process.
"""
from __future__ import annotations

import os
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .depth_generator import (
    _append_flag_if_supported,
    _append_if_supported,
    _build_depthflow_env,
    _find_depthflow_executable,
    _patch_depthflow_runtime,
    _resolve_depthflow_cli_motion,
)

logger = logging.getLogger("ShaderFlow.Modular")


def _temp_dir() -> Path:
    try:
        import folder_paths
        base = Path(folder_paths.get_temp_directory())
    except Exception:
        base = Path(tempfile.gettempdir())
    out = base / "vhs_sf_depthflow_cli"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_input(src_img: np.ndarray, width: int, height: int, key: str) -> Path:
    from PIL import Image

    img = Image.fromarray(np.clip(src_img, 0, 255).astype(np.uint8), mode="RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)

    path = _temp_dir() / f"sf_depthflow_input_{key}.png"
    img.save(path, format="PNG", compress_level=0)
    return path


def _write_depth_override(depth_np: np.ndarray | None, width: int, height: int, key: str) -> Path | None:
    if depth_np is None:
        return None

    from PIL import Image

    depth = np.asarray(depth_np, dtype=np.float32)
    if depth.shape[:2] != (height, width):
        try:
            import cv2
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass

    depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    path = _temp_dir() / f"sf_depthflow_depth_{key}.png"
    Image.fromarray((depth * 65535.0).astype(np.uint16), mode="I;16").save(path)
    return path


def _codec(params: dict) -> str:
    codec = str(params.get("video_codec") or params.get("codec") or "h264")
    if codec in {"libx264", "h264"}:
        return "h264"
    if codec in {"libx265", "h265"}:
        return "h265"
    return codec


def _build_command(
    executable: str,
    input_path: Path,
    depth_path: Path | None,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    params: dict,
) -> list[str]:
    depth_estimator = str(params.get("depth_estimator", "da2"))
    movement = str(params.get("camera_movement", params.get("movement", "zoom")))
    cli_motion, cli_caps = _resolve_depthflow_cli_motion(executable, movement)
    if depth_path is None and depth_estimator not in cli_caps["estimators"]:
        raise RuntimeError(
            f"DepthFlow CLI does not provide estimator '{depth_estimator}'. "
            "Provide a depth map or use an estimator supported by the DepthFlow venv."
        )

    command = [executable, "input", "-i", str(input_path)]
    if depth_path is None:
        command.append(depth_estimator)
    else:
        command.extend(["-d", str(depth_path)])

    if cli_motion != "static":
        command.append(cli_motion)
        movement_help = cli_caps["movement_help"].get(cli_motion, "")
        _append_if_supported(
            command,
            movement_help,
            "--intensity",
            "-i",
            str(float(params.get("movement_intensity", params.get("intensity", 1.0)))),
        )

        movement_reverse = bool(params.get("movement_reverse", False))
        movement_smooth = bool(params.get("movement_smooth", True))
        movement_loop = bool(params.get("movement_loop", True))
        movement_phase = float(params.get("movement_phase", params.get("phase", 0.0)))
        steady_depth = float(params.get("steady_depth", params.get("depth", 0.3)))
        isometric = float(params.get("isometric", 0.6))

        if cli_motion in ["vertical", "horizontal", "zoom", "dolly"]:
            if movement_reverse and not _append_flag_if_supported(command, movement_help, "--reverse"):
                _append_flag_if_supported(command, movement_help, "-r")
            if not movement_smooth and not _append_flag_if_supported(command, movement_help, "--no-smooth"):
                _append_flag_if_supported(command, movement_help, "-ns")
            if not movement_loop and not _append_flag_if_supported(command, movement_help, "--no-loop"):
                _append_flag_if_supported(command, movement_help, "-nl")
            _append_if_supported(command, movement_help, "--phase", "-p", str(movement_phase))
        elif cli_motion == "circle":
            if movement_reverse and not _append_flag_if_supported(command, movement_help, "--reverse"):
                _append_flag_if_supported(command, movement_help, "-r")
            _append_if_supported(command, movement_help, "--phase", "-p", str(movement_phase))
        elif cli_motion == "orbital" and movement_reverse:
            if not _append_flag_if_supported(command, movement_help, "--reverse"):
                _append_flag_if_supported(command, movement_help, "-r")

        if cli_motion in ["vertical", "horizontal", "circle"]:
            _append_if_supported(command, movement_help, "--steady", "-S", str(steady_depth))
        if cli_motion in ["vertical", "horizontal", "zoom", "circle"]:
            _append_if_supported(command, movement_help, "--isometric", "-I", str(isometric))
        if cli_motion in ["dolly", "orbital"]:
            _append_if_supported(command, movement_help, "--depth", "-d", str(steady_depth))

    command.append(_codec(params))
    command.extend([
        "main",
        "-w", str(int(width)),
        "-h", str(int(height)),
        "-q", str(int(float(params.get("quality_pct", 100)))),
        "-t", str(float(duration)),
        "-f", str(float(fps)),
        "-s", str(float(params.get("ssaa", 1.0))),
        "-o", str(output_path),
    ])
    return command


def init_renderer(
    layer: dict,
    src_img: np.ndarray,
    depth_np: np.ndarray | None,
    audio_depth_np: np.ndarray | None,
    width: int,
    height: int,
    fps: float,
    duration: float,
    params: dict,
) -> dict:
    import cv2

    key = f"{os.getpid()}_{id(layer) % 1000000}_{width}x{height}_{int(float(duration) * 1000)}"
    input_path = _write_input(src_img, width, height, key)
    depth_path = _write_depth_override(depth_np, width, height, key)
    output_path = _temp_dir() / f"sf_depthflow_{key}.mp4"

    executable = _find_depthflow_executable()
    _patch_depthflow_runtime(executable)
    command = _build_command(
        executable, input_path, depth_path, output_path,
        width, height, fps, duration, params,
    )

    logger.info("[SF] DepthFlow: rendering base layer through DepthFlow CLI venv")
    logger.info(f"[SF] DepthFlow CLI: {' '.join(map(str, command))}")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_build_depthflow_env(),
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"[SF DepthFlow CLI] {line}")
            tail.append(line)
            tail = tail[-40:]

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            "DepthFlow CLI OpenGL render failed.\n"
            f"cli={executable}\n"
            + "\n".join(tail)
        )
    if not output_path.is_file():
        raise RuntimeError(f"DepthFlow CLI did not produce output video: {output_path}")

    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open DepthFlow CLI output video: {output_path}")

    return {
        "cap": cap,
        "path": str(output_path),
        "next_frame": 0,
        "audio_depth": _prepare_audio_depth(audio_depth_np, width, height, params),
        "cleanup_paths": [str(input_path), str(depth_path) if depth_path is not None else ""],
    }


def _prepare_audio_depth(depth_np: np.ndarray | None, width: int, height: int, params: dict) -> np.ndarray | None:
    if depth_np is None:
        return None
    import cv2

    depth = np.asarray(depth_np, dtype=np.float32)
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    sigma = max(0.0, float(params.get("audio_depth_smooth", 1.2)))
    if sigma > 0:
        depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return depth.astype(np.float32, copy=False)


def _audio_reactive_transform(frame: np.ndarray, audio_val: float, preset: str, params: dict) -> np.ndarray:
    if audio_val <= 0.001 or preset == "none":
        return frame

    import cv2

    depth = params.get("_audio_depth")
    if depth is not None and bool(params.get("audio_depth_reactive", True)):
        return _audio_depth_warp(frame, depth, audio_val, preset, params)

    scale = 1.0
    shift_y = 0.0
    shift_x = 0.0
    rotate = 0.0
    target = params.get("audio_target", "both")

    if preset == "subtle_pulse":
        scale += audio_val * 0.035
    elif preset == "heartbeat_zoom":
        scale += audio_val * 0.12
        shift_y -= audio_val * frame.shape[0] * 0.015
    elif preset == "aggressive_bounce":
        scale += audio_val * 0.10
        shift_y -= audio_val * frame.shape[0] * 0.05
    elif preset == "chaotic_shake":
        frame_pos = float(params.get("_frame_idx", 0.0))
        scale += audio_val * 0.08
        shift_x += np.sin(frame_pos * 0.37) * audio_val * frame.shape[1] * 0.025
        shift_y += np.cos(frame_pos * 0.29) * audio_val * frame.shape[0] * 0.025
        rotate += audio_val * 0.35
    elif preset == "custom":
        if target in ("zoom", "both"):
            scale += audio_val * 0.10
        if target in ("height", "both"):
            shift_y -= audio_val * frame.shape[0] * 0.04
        if target == "isometric":
            rotate += audio_val * 0.25
        if target == "phase":
            shift_x += audio_val * frame.shape[1] * 0.04

    if abs(scale - 1.0) < 1e-5 and abs(shift_x) < 1e-5 and abs(shift_y) < 1e-5 and abs(rotate) < 1e-5:
        return frame

    h, w = frame.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), rotate, scale)
    mat[0, 2] += shift_x
    mat[1, 2] += shift_y
    return cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _audio_depth_warp(frame: np.ndarray, depth: np.ndarray, audio_val: float, preset: str, params: dict) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    strength = float(params.get("audio_depth_strength", 1.0))
    max_px = max(0.0, float(params.get("audio_depth_max_px", 18.0)))
    if strength <= 0.0 or max_px <= 0.0:
        return frame

    target = params.get("audio_target", "both")
    mode = str(params.get("audio_depth_mode", "background_only"))
    near_weight = float(params.get("audio_depth_near_weight", 0.15))
    far_weight = float(params.get("audio_depth_far_weight", 1.0))
    near_protect = float(params.get("audio_depth_near_protect", 0.65))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * float(params.get("audio_depth_center_x", 0.5))
    cy = (h - 1) * float(params.get("audio_depth_center_y", 0.5))
    max_dim = float(max(w, h))

    z01 = np.clip(depth.astype(np.float32), 0.0, 1.0)
    central = z01[h // 4:max(h // 4 + 1, h * 3 // 4), w // 4:max(w // 4 + 1, w * 3 // 4)]
    center_median = float(np.median(central)) if central.size else float(np.median(z01))
    global_median = float(np.median(z01))
    near_raw = z01 if center_median >= global_median else (1.0 - z01)
    near_raw = np.clip(near_raw, 0.0, 1.0)
    start = max(0.0, min(0.98, near_protect - 0.25))
    end = max(start + 0.01, min(1.0, near_protect))
    near_score = np.clip((near_raw - start) / (end - start), 0.0, 1.0)
    near_score = near_score * near_score * (3.0 - 2.0 * near_score)

    if mode == "full_scene":
        influence = np.ones_like(z01, dtype=np.float32)
    elif mode == "foreground_only":
        influence = near_score * max(0.0, near_weight)
    elif mode == "balanced":
        influence = near_score * max(0.0, near_weight) + (1.0 - near_score) * max(0.0, far_weight)
    else:
        influence = (1.0 - near_score) * max(0.0, far_weight) + near_score * max(0.0, near_weight)

    z = (z01 - float(z01.mean())) * 2.0 * influence
    radial_x = (xx - cx) / max_dim
    radial_y = (yy - cy) / max_dim

    zoom_amt = 0.0
    height_amt = 0.0
    dolly_amt = 0.0
    phase_amt = 0.0
    frame_pos = float(params.get("_frame_idx", 0.0))

    if preset == "subtle_pulse":
        zoom_amt = audio_val * 28.0 * strength
        height_amt = audio_val * 8.0 * strength
    elif preset == "heartbeat_zoom":
        zoom_amt = audio_val * 70.0 * strength
        dolly_amt = audio_val * 28.0 * strength
    elif preset == "aggressive_bounce":
        zoom_amt = audio_val * 95.0 * strength
        height_amt = audio_val * 45.0 * strength
        dolly_amt = audio_val * 40.0 * strength
    elif preset == "chaotic_shake":
        zoom_amt = audio_val * 55.0 * strength
        phase_amt = audio_val * 35.0 * strength
        height_amt = np.cos(frame_pos * 0.31) * audio_val * 28.0 * strength
    elif preset == "custom":
        if target in ("zoom", "both"):
            zoom_amt = audio_val * 75.0 * strength
        if target in ("height", "both"):
            height_amt = audio_val * 45.0 * strength
        if target == "isometric":
            dolly_amt = audio_val * 55.0 * strength
        if target == "phase":
            phase_amt = audio_val * 45.0 * strength

    dx = radial_x * z * zoom_amt
    dy = radial_y * z * zoom_amt
    dy -= z * height_amt
    dx += z * dolly_amt * 0.35
    if phase_amt:
        dx += np.sin(frame_pos * 0.17 + z * 2.5) * phase_amt * z
        dy += np.cos(frame_pos * 0.13 + z * 2.0) * phase_amt * z * 0.5

    mag = np.maximum(1.0, np.sqrt(dx * dx + dy * dy) / max_px)
    dx = dx / mag
    dy = dy / mag

    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def render_frame(
    renderer: dict,
    width: int,
    height: int,
    frame_idx: int,
    audio_val: float = 0.0,
    preset: str = "none",
    params: dict | None = None,
) -> np.ndarray:
    import cv2

    cap = renderer["cap"]
    if renderer.get("next_frame", 0) != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

    ok, frame_bgr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx) - 1))
        ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read DepthFlow CLI frame {frame_idx} from {renderer.get('path')}")

    renderer["next_frame"] = frame_idx + 1
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if params is None:
        params = {}
    params["_frame_idx"] = frame_idx
    params["_audio_depth"] = renderer.get("audio_depth")
    frame = _audio_reactive_transform(frame, float(audio_val), preset, params)
    return frame.astype(np.uint8, copy=False)
