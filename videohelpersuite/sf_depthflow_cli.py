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
        "cleanup_paths": [str(input_path), str(depth_path) if depth_path is not None else ""],
    }


def _audio_reactive_transform(frame: np.ndarray, audio_val: float, preset: str, params: dict) -> np.ndarray:
    if audio_val <= 0.001 or preset == "none":
        return frame

    import cv2

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
    frame = _audio_reactive_transform(frame, float(audio_val), preset, params)
    return frame.astype(np.uint8, copy=False)
