"""
ShaderFlow Modular Core — Type definitions, logging, layer rendering.

All SF_* types are lightweight dicts (recipes/descriptors).
Actual rendering happens only in SF_RenderToVideo via render_layer_frame().
"""
import math
import logging
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
T_SF_PIPELINE = "SF_PIPELINE"


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
    elif layer_type == "depthflow":
        return _render_depthflow_frame(layer, frame_idx, t, w, h, fps, canvas)
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

    bg_img = layer.get("background_frame")
    if bg_img is not None and callable(bg_img):
        frame = bg_img(frame_idx, w, h).copy()
    else:
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

    bg_img = layer.get("background_frame")
    if bg_img is not None and callable(bg_img):
        frame = bg_img(frame_idx, w, h).copy()
    else:
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
# DepthFlow layer renderer
# ---------------------------------------------------------------------------


def _setup_env_paths():
    """Inject DepthFlow and ShaderFlow virtual environments into sys.path and setup OpenGL env."""
    import sys
    import os
    import platform
    import importlib
    from pathlib import Path
    
    # --- 1. Linux EGL Environment Setup ---
    if platform.system() != "Windows":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        
        # Check NVIDIA GPU
        has_nvidia = False
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "NVIDIA" in r.stdout:
                has_nvidia = True
        except Exception:
            pass
            
        if has_nvidia:
            # Find ICD
            egl_icd = None
            candidates = [
                "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
                "/usr/share/egl/egl_external_platform.d/10_nvidia.json",
                "/etc/glvnd/egl_vendor.d/10_nvidia.json",
                "/usr/lib/x86_64-linux-gnu/egl_vendor.d/10_nvidia.json",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    egl_icd = c
                    break
            
            if egl_icd:
                os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_icd
            os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
            os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
            os.environ.pop("MESA_GL_VERSION_OVERRIDE", None)
        else:
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")
            os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")

    # --- 2. Virtual Environment Paths ---
    node_dir = Path(__file__).resolve().parent.parent # ComfyUI-VideoHelperSuite
    base_dir = node_dir.parent.parent                 # ComfyUI (or ComfyUI/own)
    
    path_modified = False
    
    # Check node-specific isolated venvs first
    isolated_venvs = [
        node_dir / ".venv_depthflow",
        node_dir / ".venv_shaderflow"
    ]
    for venv_path in isolated_venvs:
        if venv_path.is_dir():
            venv_libs = list(venv_path.glob("lib/python*/site-packages"))
            if venv_libs and str(venv_libs[0]) not in sys.path:
                sys.path.insert(0, str(venv_libs[0]))
                path_modified = True

    # Check global parallel directories as fallbacks
    projects = ["DepthFlow", "ShaderFlow"]
    for proj in projects:
        proj_dir = base_dir / proj
        if proj_dir.is_dir():
            venv_lib = list(proj_dir.glob('.venv*/lib/python*/site-packages'))
            if venv_lib and str(venv_lib[0]) not in sys.path:
                sys.path.insert(0, str(venv_lib[0]))
                path_modified = True
            
            if str(proj_dir) not in sys.path:
                sys.path.insert(0, str(proj_dir))
                path_modified = True

    # --- 3. Force Reload Outdated Modules ---
    if path_modified:
        # If ComfyUI loaded an old 'attr' from system Python, force reload it from the new venv
        for mod_name in ["attr", "attrs"]:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass


def _estimate_depth_simple(img_np, estimator: str = "da2"):
    """Estimate depth map. Tries HF pipeline, falls back to luminance."""
    import numpy as np
    import logging
    logger = logging.getLogger("ShaderFlow.Modular")
    try:
        from transformers import pipeline as hf_pipeline
        model_ids = {
            "da2": "depth-anything/Depth-Anything-V2-Small-hf",
            "da1": "LiheYoung/depth-anything-small-hf",
        }
        model_id = model_ids.get(estimator, model_ids["da2"])
        logger.info(f"[SF] Estimating depth with {model_id}...")
        pipe = hf_pipeline("depth-estimation", model=model_id, device=0)
        from PIL import Image as _PIL
        result = pipe(_PIL.fromarray(img_np))
        depth = np.array(result["depth"], dtype=np.float32)
        if depth.max() > 0:
            depth = depth / depth.max()
        return depth
    except Exception as e:
        logger.warning(f"[SF] HF depth failed ({e}), using luminance fallback")
        gray = np.mean(img_np.astype(np.float32), axis=2)
        return 1.0 - (gray / max(gray.max(), 1.0))


def _render_depthflow_frame(layer: dict, frame_idx: int, t: float,
                            w: int, h: int, fps: float,
                            canvas: dict):
    """DepthFlow frame-by-frame O(1) memory rendering with audio reactivity."""
    params = layer.get("params", {})
    src_img = layer["_src_img"]
    depth_np = layer.get("_depth_np")
    total = get_total_frames(canvas)
    duration = canvas.get("duration", total / max(fps, 1.0))
    import numpy as np

    # Ensure shape alignment (Sanity Check)
    if depth_np is None:
        depth_np = _estimate_depth_simple(src_img, params.get("depth_estimator", "da2"))
        
    if depth_np.shape[:2] != src_img.shape[:2]:
        try:
            import cv2
            depth_np = cv2.resize(depth_np, (src_img.shape[1], src_img.shape[0]))
        except ImportError:
            pass
    layer["_depth_np"] = depth_np

    # Initialize renderer fallback chain
    if layer.get("_renderer") is None:
        logger.info("[SF] DepthFlow: checking rendering backends...")
        img_f = src_img.astype(np.float32) / 255.0
        dep_f = depth_np.astype(np.float32)
        
        renderer = None
        cuda_mod = None
        
        # 1. Try Native ShaderFlow OpenGL (with full lifecycle)
        try:
            _setup_env_paths()
            import depthflow
            from depthflow.scene import DepthScene
            from shaderflow.scene import WindowBackend
            from shaderflow.message import ShaderMessage
            
            scene = DepthScene(backend=WindowBackend.Headless)
            # Set raw data before initialize so _load_inputs works
            scene._raw_image = src_img
            scene._raw_depth = depth_np
            
            # Exact initialization sequence identical to ShaderScene.main()
            scene.initialize()
            scene.relay(ShaderMessage.Shader.Compile)
            for module in scene.modules:
                module.setup()
                
            renderer = ("opengl", scene)
            logger.info("[SF] DepthFlow: Native OpenGL initialized successfully")
        except Exception as e:
            logger.info(f"[SF] DepthFlow: Native OpenGL unavailable ({e}), trying CUDA...")
        
        # 2. Try CUDA — DepthFlow's official non-GUI renderer (same quality as OpenGL)
        if renderer is None:
            from .df_nodes_pipeline import _find_cuda_renderer
            cuda_mod = _find_cuda_renderer()
            if cuda_mod is not None and cuda_mod.is_available():
                renderer = ("cuda", cuda_mod.CudaDepthFlowRenderer(img_f, dep_f))
                logger.info("[SF] DepthFlow: CUDA renderer initialized (official quality)")
            else:
                logger.warning("[SF] DepthFlow: CUDA unavailable, using CPU fallback")
                renderer = ("cpu", None)
                
        layer["_renderer"] = renderer
        layer["_cuda_mod"] = cuda_mod

    renderer_type, renderer_obj = layer["_renderer"]
    cuda_mod = layer["_cuda_mod"]

    audio_val = 0.0
    preset = params.get("audio_preset", "none")
    if preset != "none" and "audio_rms" in canvas:
        rms_arr = canvas["audio_rms"]
        if len(rms_arr) > 0:
            audio_val = rms_arr[min(frame_idx, len(rms_arr)-1)]
            audio_val *= params.get("audio_scale", 1.5)

    if renderer_type == "opengl":
        return _render_df_frame_opengl(renderer_obj, w, h, total, frame_idx, params, audio_val, preset)
    elif renderer_type == "cuda" and cuda_mod is not None:
        return _render_df_frame_cuda(renderer_obj, cuda_mod, w, h, total, frame_idx, params, audio_val, preset)
    else:
        return _render_df_frame_fallback(src_img, depth_np, w, h, total, frame_idx, params, audio_val, preset)


def _apply_audio_preset(state, audio_val: float, preset: str, params: dict):
    """Apply highly customizable audio reactivity to the DepthFlow state."""
    if audio_val <= 0.001 or preset == "none":
        return

    if preset == "subtle_pulse":
        state.zoom += audio_val * 0.05
    elif preset == "heartbeat_zoom":
        state.zoom += audio_val * 0.2
        state.height += audio_val * 0.1
    elif preset == "aggressive_bounce":
        state.zoom += audio_val * 0.15
        state.height += audio_val * 0.3
        state.dolly += audio_val * 0.1
    elif preset == "chaotic_shake":
        state.zoom += audio_val * 0.2
        state.height += audio_val * 0.3
        state.dolly -= audio_val * 0.15
        state.isometric += audio_val * 0.2
    elif preset == "custom":
        target = params.get("audio_target", "both")
        if target in ("zoom", "both"):
            state.zoom += audio_val * 0.15
        if target in ("height", "both"):
            state.height += audio_val * 0.25
        if target == "isometric":
            state.isometric += audio_val * 0.3
        if target == "phase":
            state.phase += audio_val * 0.5


def _render_df_frame_opengl(scene, w, h, total, frame_idx, params, audio_val: float, preset: str):
    import numpy as np
    tau = frame_idx / max(total - 1, 1)
    
    # Map params to scene.state
    # Native ShaderFlow updating requires setting scene.tau or using step
    # We will manually construct state
    try:
        from depthflow.state import DepthState
        import depthflow.animation as dfa
    except ImportError:
        return np.zeros((h, w, 3), dtype=np.uint8)

    state = DepthState()
    
    # Map params to scene.state
    move_map = {
        "vertical": getattr(dfa, "Vertical", None),
        "horizontal": getattr(dfa, "Horizontal", None),
        "circle": getattr(dfa, "Circle", None),
        "zoom": getattr(dfa, "Zoom", None),
        "dolly": getattr(dfa, "Dolly", None),
        "orbital": getattr(dfa, "Orbital", None),
    }
    move_cls = move_map.get(params.get("camera_movement", "vertical"))
    
    if move_cls:
        # 1. Initialize native DepthFlow implementation
        action = move_cls()
        
        # 2. Map properties safely to the new Pydantic schema (nested 'Sine' wave)
        intensity = params.get("movement_intensity", 1.0)
        phase = params.get("movement_phase", 0.0)
        cycles = 1.0 if params.get("movement_loop", True) else 0.5
        
        if hasattr(action, 'wave'):
            action.wave.amplitude = intensity
            action.wave.phase = phase
            action.wave.cycles = cycles
        elif hasattr(action, 'x') and hasattr(action, 'y'):
            action.x.amplitude = intensity
            action.x.phase = phase
            action.x.cycles = cycles
            action.y.amplitude = intensity
            action.y.phase = phase
            action.y.cycles = cycles
        else:
            # Fallback for old/other native implementations
            if hasattr(action, 'intensity'): action.intensity = intensity
            if hasattr(action, 'phase'): action.phase = phase
            if hasattr(action, 'loop'): action.loop = params.get("movement_loop", True)
            if hasattr(action, 'reverse'): action.reverse = params.get("movement_reverse", False)
            if hasattr(action, 'smooth'): action.smooth = params.get("movement_smooth", True)
            if hasattr(action, 'steady'): action.steady = params.get("steady_depth", 0.3)
            if hasattr(action, 'isometric'): action.isometric = params.get("isometric", 0.6)
            
        try:
            action.apply(state, tau)
        except TypeError as e:
            # Workaround for DepthFlow native bug where Vertical/Horizontal try to mutate a tuple:
            # e.g., state.offset[1] = self.wave.at(time)
            if "tuple" in str(e):
                if move_cls.__name__ == "Vertical":
                    state.offset = (state.offset[0], action.wave.at(tau))
                elif move_cls.__name__ == "Horizontal":
                    state.offset = (action.wave.at(tau), state.offset[1])
                else:
                    raise e
            else:
                raise e
        
        # 3. Global parameters not mutated by specific native offset Actions
        state.steady = params.get("steady_depth", 0.3)
        state.isometric = params.get("isometric", 0.6)
        
    else:
        # Only fallback to CUDA math if the native implementation class is MISSING
        try:
            from depthflow.cuda_renderer import compute_animation_state
            c_state = compute_animation_state(
                params.get("camera_movement", "vertical"), tau,
                intensity=params.get("movement_intensity", 1.0),
                smooth=params.get("movement_smooth", True),
                loop=params.get("movement_loop", True),
                reverse=params.get("movement_reverse", False),
                phase=params.get("movement_phase", 0.0),
                steady_depth=params.get("steady_depth", 0.3),
                isometric=params.get("isometric", 0.6)
            )
            state.height = c_state.height
            state.steady = c_state.steady
            state.focus = c_state.focus
            state.zoom = c_state.zoom
            state.isometric = c_state.isometric
            state.dolly = c_state.dolly
            state.offset = (c_state.offset_x, c_state.offset_y)
        except ImportError:
            state.steady = params.get("steady_depth", 0.3)
            state.isometric = params.get("isometric", 0.6)

    # Apply Audio Presets
    _apply_audio_preset(state, audio_val, preset, params)

    scene.state = state
    # Resize resolution by explicitly using keyword args to bypass upstream `resolution` setter unpacking bug
    scene.resize(width=w, height=h)
    scene.next(0.0)
    
    frame_np = scene.screenshot()
    return frame_np


def _render_df_frame_cuda(renderer, cuda_mod, w, h, total, frame_idx, params, audio_val: float, preset: str):
    import numpy as np
    tau = frame_idx / max(total - 1, 1)
    
    state = cuda_mod.compute_animation_state(
        params.get("camera_movement", "vertical"), tau,
        intensity=params.get("movement_intensity", 1.0),
        smooth=params.get("movement_smooth", True),
        loop=params.get("movement_loop", True),
        reverse=params.get("movement_reverse", False),
        phase=params.get("movement_phase", 0.0),
        steady_depth=params.get("steady_depth", 0.3),
        isometric=params.get("isometric", 0.6)
    )
    
    _apply_audio_preset(state, audio_val, preset, params)

    ssaa = params.get("ssaa", 1.0)
    ssaa_w = int(w * ssaa)
    ssaa_h = int(h * ssaa)
    
    eff_aa = True if ssaa <= 1.0 else False
    frame_tensor = renderer.render_frame(
        ssaa_w, ssaa_h, state, quality_pct=90,
        enable_inpaint=True, enable_aa=eff_aa
    )
    
    # render_frame() returns (H,W,3) uint8 tensor — DO NOT multiply by 255!
    frame_np = frame_tensor.numpy()
    if ssaa != 1.0:
        try:
            import cv2
            frame_np = cv2.resize(frame_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        except ImportError:
            pass
            
    return frame_np


def _render_df_frame_fallback(src_img, depth_np, w, h, total, frame_idx, params, audio_val: float, preset: str):
    """CPU-only parallax simulation using vectorized cv2.remap. O(1) per frame."""
    import numpy as np
    import math
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    intensity = params.get("movement_intensity", 1.0)
    movement = params.get("camera_movement", "vertical")
    loop = params.get("movement_loop", True)

    # Resize source and depth to output size
    if has_cv2:
        src = cv2.resize(src_img, (w, h))
        depth = cv2.resize(depth_np, (w, h)).astype(np.float32)
    else:
        # No cv2 = can't do remap, return scaled source
        out = np.zeros((h, w, 3), dtype=np.uint8)
        rh, rw = min(src_img.shape[0], h), min(src_img.shape[1], w)
        out[:rh, :rw] = src_img[:rh, :rw]
        return out

    # Normalize depth to 0..1
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    else:
        depth = np.zeros_like(depth)

    # Compute phase animation
    phase = frame_idx / max(total - 1, 1)
    if loop:
        t_val = math.sin(phase * math.pi * 2) * intensity
    else:
        t_val = (phase * 2 - 1) * intensity

    # Audio reactivity
    if preset != "none" and audio_val > 0.01:
        scale_map = {
            "subtle_pulse": 0.3, "heartbeat_zoom": 0.8,
            "aggressive_bounce": 1.5, "chaotic_shake": 2.0, "custom": 1.0,
        }
        t_val += audio_val * scale_map.get(preset, 1.0)

    # Build displacement maps using vectorized operations
    # Base coordinate grids
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    cx, cy = w / 2.0, h / 2.0
    max_shift = min(w, h) * 0.05  # max pixel displacement as fraction of size

    if movement == "vertical":
        # Shift Y based on depth: deeper pixels shift more
        shift = depth * t_val * max_shift
        map_x = grid_x
        map_y = grid_y - shift

    elif movement == "horizontal":
        shift = depth * t_val * max_shift
        map_x = grid_x - shift
        map_y = grid_y

    elif movement == "circle":
        angle = phase * math.pi * 2
        dx = math.cos(angle) * intensity * max_shift
        dy = math.sin(angle) * intensity * max_shift
        shift_x = depth * dx
        shift_y = depth * dy
        if preset != "none" and audio_val > 0.01:
            shift_x *= (1.0 + audio_val * 0.5)
            shift_y *= (1.0 + audio_val * 0.5)
        map_x = grid_x - shift_x
        map_y = grid_y - shift_y

    elif movement == "zoom":
        # Zoom: scale from center based on depth
        scale = 1.0 + depth * t_val * 0.1
        map_x = cx + (grid_x - cx) / scale
        map_y = cy + (grid_y - cy) / scale

    elif movement == "dolly":
        # Dolly: forward/backward motion based on depth
        scale = 1.0 + depth * t_val * 0.08
        map_x = cx + (grid_x - cx) / scale
        map_y = cy + (grid_y - cy) / scale

    elif movement == "orbital":
        angle = phase * math.pi * 2
        dx = math.cos(angle) * intensity * max_shift * 0.7
        dy = math.sin(angle) * intensity * max_shift * 0.3
        shift_x = depth * dx
        shift_y = depth * dy
        map_x = grid_x - shift_x
        map_y = grid_y - shift_y

    else:
        # Default: vertical
        shift = depth * t_val * max_shift
        map_x = grid_x
        map_y = grid_y - shift

    # Apply remap with border replication (prevents black edges)
    frame = cv2.remap(
        src, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return frame




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

