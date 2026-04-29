"""
ShaderFlow Modular Core — Type definitions, logging, layer rendering.

All SF_* types are lightweight dicts (recipes/descriptors).
Actual rendering happens only in SF_RenderToVideo via render_layer_frame().
"""
import os
import math
import logging
import hashlib
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ShaderFlow.Modular")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Custom ComfyUI type names (string constants)
# ---------------------------------------------------------------------------
T_SF_AUDIO = "SF_AUDIO"
T_SF_SPECTRUM = "SF_SPECTRUM"
T_SF_CURVE = "SF_CURVE"
T_SF_MIDI = "SF_MIDI"
T_SF_CANVAS = "SF_CANVAS"
T_SF_LAYER = "SF_LAYER"


# ---------------------------------------------------------------------------
# Layer rendering dispatch
# ---------------------------------------------------------------------------

def render_layer_frame(
    layer: Dict[str, Any],
    frame_idx: int,
    canvas: Dict[str, Any],
) -> np.ndarray:
    """Render a single frame for a given SF_LAYER descriptor.
    Returns (H, W, 3) uint8 ndarray.
    This is called per-frame by the final render node — never holds all frames in memory.
    """
    w = canvas["width"]
    h = canvas["height"]
    fps = canvas["fps"]
    t = frame_idx / fps
    dt = 1.0 / fps
    layer_type = layer.get("type", "solid")

    if layer_type == "bars":
        return _render_bars_frame(layer, frame_idx, t, w, h)
    elif layer_type == "radial":
        return _render_radial_frame(layer, frame_idx, t, w, h)
    elif layer_type == "waveform":
        return _render_waveform_frame(layer, frame_idx, t, w, h)
    elif layer_type == "piano_roll":
        return _render_piano_roll_frame(layer, frame_idx, t, w, h)
    elif layer_type == "glsl":
        return _render_glsl_frame(layer, frame_idx, t, dt, w, h)
    elif layer_type == "color_grade":
        return _apply_color_grade_frame(layer, frame_idx)
    elif layer_type == "motion_blur":
        return _apply_motion_blur_frame(layer, frame_idx)
    else:
        return np.zeros((h, w, 3), dtype=np.uint8)


def _get_spectrum_frame(layer: Dict, frame_idx: int) -> np.ndarray:
    """Extract spectrum bins for a specific frame from SF_SPECTRUM embedded in layer."""
    spectrum = layer.get("spectrum")
    if spectrum is None:
        return np.zeros(64, dtype=np.float32)
    bins_data = spectrum["bins"]
    idx = min(frame_idx, len(bins_data) - 1)
    return bins_data[idx]


# ---------------------------------------------------------------------------
# Bars renderer
# ---------------------------------------------------------------------------

def _render_bars_frame(layer: Dict, frame_idx: int, t: float, w: int, h: int) -> np.ndarray:
    spec = _get_spectrum_frame(layer, frame_idx)
    params = layer.get("params", {})
    mirror = params.get("mirror", True)
    glow = params.get("glow", True)
    bg_color = tuple(params.get("bg_color", (10, 10, 25)))

    colors = params.get("colors", [
        (0, 255, 220), (120, 80, 255), (255, 50, 180), (255, 160, 40),
    ])

    # Composite with background if provided
    bg_img = layer.get("background_frame")
    if bg_img is not None and callable(bg_img):
        frame = bg_img(frame_idx, w, h)
    else:
        frame = np.full((h, w, 3), bg_color, dtype=np.uint8)

    bins = len(spec)
    if bins == 0:
        return frame

    bar_area_w = int(w * 0.9)
    bar_area_x = (w - bar_area_w) // 2
    bar_w = max(1, bar_area_w // bins)
    gap = max(0, (bar_area_w - bar_w * bins) // max(1, bins - 1))
    actual_bar_w = max(1, bar_w - 1)
    max_bar_h = int(h * 0.40)
    center_y = h // 2

    for i in range(bins):
        val = float(np.clip(spec[i], 0, 1))
        bh = int(val * max_bar_h)
        if bh < 1:
            continue
        x = bar_area_x + i * (bar_w + gap)
        if x + actual_bar_w > w:
            break
        color = _gradient_color(i / max(1, bins - 1), colors)
        y_top = center_y - bh
        frame[max(0, y_top):center_y, x:x + actual_bar_w] = color
        if mirror:
            y_bot = min(center_y + bh, h)
            frame[center_y:y_bot, x:x + actual_bar_w] = color

    if glow:
        try:
            import cv2
            blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=15, sigmaY=15)
            frame = np.clip(
                frame.astype(np.float32) + blurred.astype(np.float32) * 0.4,
                0, 255
            ).astype(np.uint8)
        except ImportError:
            pass
    return frame


# ---------------------------------------------------------------------------
# Radial renderer
# ---------------------------------------------------------------------------

def _render_radial_frame(layer: Dict, frame_idx: int, t: float, w: int, h: int) -> np.ndarray:
    spec = _get_spectrum_frame(layer, frame_idx)
    params = layer.get("params", {})
    bg_color = tuple(params.get("bg_color", (10, 10, 25)))
    colors = params.get("colors", [
        (0, 255, 220), (120, 80, 255), (255, 50, 180), (255, 160, 40),
    ])

    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    bins = len(spec)
    if bins == 0:
        return frame

    try:
        import cv2
    except ImportError:
        return frame

    cx, cy = w // 2, h // 2
    r_inner = min(cx, cy) * 0.2
    r_max = min(cx, cy) * 0.7

    for i in range(bins):
        angle = 2.0 * math.pi * i / bins - math.pi / 2
        val = float(np.clip(spec[i], 0, 1))
        r_outer = r_inner + val * (r_max - r_inner)
        x1 = int(cx + r_inner * math.cos(angle))
        y1 = int(cy + r_inner * math.sin(angle))
        x2 = int(cx + r_outer * math.cos(angle))
        y2 = int(cy + r_outer * math.sin(angle))
        color = _gradient_color(i / max(1, bins - 1), colors)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=12)
    frame = np.clip(
        frame.astype(np.float32) + blurred.astype(np.float32) * 0.3,
        0, 255
    ).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Waveform renderer
# ---------------------------------------------------------------------------

def _render_waveform_frame(layer: Dict, frame_idx: int, t: float, w: int, h: int) -> np.ndarray:
    params = layer.get("params", {})
    bg_color = tuple(params.get("bg_color", (10, 10, 25)))
    line_color = tuple(params.get("line_color", (0, 255, 180)))
    audio = layer.get("audio")

    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    if audio is None:
        return frame

    try:
        import cv2
    except ImportError:
        return frame

    samples = audio["samples"]
    sr = audio["sr"]
    # Extract ~1/fps seconds of waveform centered at t
    fps = layer.get("fps", 30.0)
    window_samples = int(sr / fps)
    center = int(t * sr)
    start_s = max(0, center - window_samples // 2)
    end_s = min(samples.shape[1], start_s + window_samples)
    chunk = samples[0, start_s:end_s] if samples.ndim == 2 else samples[start_s:end_s]

    if len(chunk) == 0:
        return frame

    center_y = h // 2
    amp = h * 0.35
    n_points = min(len(chunk), w)
    indices = np.linspace(0, len(chunk) - 1, n_points).astype(int)
    sampled = chunk[indices]

    points = np.zeros((n_points, 1, 2), dtype=np.int32)
    for i in range(n_points):
        px = int(i * w / n_points)
        py = int(np.clip(center_y - sampled[i] * amp, 0, h - 1))
        points[i, 0] = [px, py]

    cv2.polylines(frame, [points], False, line_color, 2, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=8)
    frame = np.clip(
        frame.astype(np.float32) + blurred.astype(np.float32) * 0.5,
        0, 255
    ).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Piano roll renderer
# ---------------------------------------------------------------------------

def _render_piano_roll_frame(layer: Dict, frame_idx: int, t: float, w: int, h: int) -> np.ndarray:
    from .shaderflow_bridge import PianoRollRenderer, DynamicNumber
    params = layer.get("params", {})
    midi = layer.get("midi")
    if midi is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    renderer = PianoRollRenderer(width=w, height=h)
    notes = midi["notes"]
    min_note = midi.get("min_note", 21)
    max_note = midi.get("max_note", 108)
    roll_window = params.get("roll_window", 3.0)
    piano_height = params.get("piano_height", 0.15)

    # Key dynamics from pre-computed cache if available
    key_dynamics = layer.get("_key_dynamics_cache")
    kd_array = None
    if key_dynamics is not None and frame_idx < len(key_dynamics):
        kd_array = key_dynamics[frame_idx]

    return renderer.render_frame(
        time=t, notes=notes, roll_time=roll_window,
        min_note=min_note, max_note=max_note,
        piano_height_ratio=piano_height,
        key_dynamics=kd_array,
    )


# ---------------------------------------------------------------------------
# GLSL renderer
# ---------------------------------------------------------------------------

def _render_glsl_frame(layer: Dict, frame_idx: int, t: float, dt: float, w: int, h: int) -> np.ndarray:
    glsl_renderer = layer.get("_glsl_renderer")
    if glsl_renderer is None:
        # Fallback: procedural rainbow
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        uv_x, uv_y = xx / w, yy / h
        r = (0.5 + 0.5 * np.cos(t + uv_x * 6.28)) * 255
        g = (0.5 + 0.5 * np.cos(t + uv_y * 6.28 + 2.0)) * 255
        b = (0.5 + 0.5 * np.cos(t + (uv_x + uv_y) * 3.14 + 4.0)) * 255
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    result = glsl_renderer.render_frame(time=t, frame=frame_idx, dt=dt)
    if result is not None:
        return result
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Post-processing layer renderers (operate on previous layer output)
# ---------------------------------------------------------------------------

def _apply_color_grade_frame(layer: Dict, frame_idx: int) -> np.ndarray:
    """Applied as post-process — expects 'source_layer' and 'canvas' in layer."""
    source = layer.get("source_layer")
    canvas = layer.get("canvas")
    if source is None or canvas is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    frame = render_layer_frame(source, frame_idx, canvas)
    params = layer.get("params", {})
    img = frame.astype(np.float32) / 255.0

    exposure = params.get("exposure", 0.0)
    if exposure != 0.0:
        img = img * (2.0 ** exposure)

    temp = params.get("temperature", 0.0)
    tint = params.get("tint", 0.0)
    if temp != 0.0 or tint != 0.0:
        img[:, :, 0] += temp * 0.1
        img[:, :, 1] -= tint * 0.05
        img[:, :, 2] -= temp * 0.1

    contrast = params.get("contrast", 1.0)
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5

    saturation = params.get("saturation", 1.0)
    if saturation != 1.0:
        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        luma = luma[:, :, np.newaxis]
        img = luma + (img - luma) * saturation

    gamma = params.get("gamma", 1.0)
    if gamma != 1.0:
        img = np.clip(img, 0, None) ** (1.0 / gamma)

    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _apply_motion_blur_frame(layer: Dict, frame_idx: int) -> np.ndarray:
    """Motion blur — needs access to previous frame accumulator."""
    source = layer.get("source_layer")
    canvas = layer.get("canvas")
    if source is None or canvas is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    frame = render_layer_frame(source, frame_idx, canvas)
    strength = layer.get("params", {}).get("strength", 0.3)

    accum = layer.get("_accum")
    if accum is None or accum.shape != frame.shape:
        layer["_accum"] = frame.astype(np.float32)
        return frame

    accum_f = layer["_accum"]
    blended = accum_f * strength + frame.astype(np.float32) * (1.0 - strength)
    layer["_accum"] = blended
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _gradient_color(t: float, colors: list) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    n = len(colors)
    if n <= 1:
        return tuple(colors[0]) if colors else (255, 255, 255)
    idx = t * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return tuple(int(colors[lo][c] * (1 - frac) + colors[hi][c] * frac) for c in range(3))


def get_total_frames(canvas: Dict[str, Any]) -> int:
    return max(1, int(canvas["duration"] * canvas["fps"]))
