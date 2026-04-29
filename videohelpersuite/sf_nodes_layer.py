"""
ShaderFlow Modular Nodes — Layer descriptor nodes.
These nodes output SF_LAYER (lazy recipe) — NO frames are rendered here.
Actual rendering happens only in SF_RenderToVideo.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .sf_core import (
    T_SF_AUDIO, T_SF_SPECTRUM, T_SF_CURVE, T_SF_MIDI,
    T_SF_CANVAS, T_SF_LAYER,
    logger,
)
from .shaderflow_bridge import DynamicNumber

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Helper: convert IMAGE tensor to background frame getter
# ---------------------------------------------------------------------------

def _make_bg_getter(image_tensor):
    """Create a callable that returns resized background frame for a given index."""
    if image_tensor is None:
        return None

    if HAS_TORCH and isinstance(image_tensor, torch.Tensor):
        arr = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    else:
        arr = (np.asarray(image_tensor) * 255).astype(np.uint8)

    # arr shape: (N, H, W, 3) or (H, W, 3)
    if arr.ndim == 3:
        arr = arr[np.newaxis, :]

    n_frames = arr.shape[0]

    def getter(frame_idx: int, w: int, h: int) -> np.ndarray:
        idx = min(frame_idx, n_frames - 1)
        frame = arr[idx]
        if frame.shape[0] != h or frame.shape[1] != w:
            try:
                import cv2
                frame = cv2.resize(frame, (w, h))
            except ImportError:
                pass
        return frame

    return getter


# ---------------------------------------------------------------------------
# SF_LayerBars — spectrum bars visualization layer
# ---------------------------------------------------------------------------

class SF_LayerBars:
    """Create bars visualization layer descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_spectrum": (T_SF_SPECTRUM,),
                "mirror": ("BOOLEAN", {"default": True, "tooltip": "上下镜像"}),
                "glow": ("BOOLEAN", {"default": True, "tooltip": "发光效果"}),
                "color_theme": (["neon", "fire", "ice", "rainbow", "mono_cyan", "mono_magenta"], {
                    "default": "neon",
                }),
            },
            "optional": {
                "background": ("IMAGE", {"tooltip": "可选背景图像"}),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    COLOR_THEMES = {
        "neon": [(0, 255, 220), (120, 80, 255), (255, 50, 180), (255, 160, 40)],
        "fire": [(255, 50, 0), (255, 150, 0), (255, 220, 50), (255, 255, 180)],
        "ice": [(0, 100, 255), (0, 200, 255), (150, 230, 255), (220, 245, 255)],
        "rainbow": [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128)],
        "mono_cyan": [(0, 80, 100), (0, 200, 220), (0, 255, 255)],
        "mono_magenta": [(80, 0, 60), (200, 0, 150), (255, 50, 200)],
    }

    def create(self, sf_spectrum, mirror, glow, color_theme, background=None):
        colors = self.COLOR_THEMES.get(color_theme, self.COLOR_THEMES["neon"])
        layer = {
            "type": "bars",
            "spectrum": sf_spectrum,
            "params": {
                "mirror": mirror,
                "glow": glow,
                "colors": colors,
                "bg_color": (10, 10, 25),
            },
            "background_frame": _make_bg_getter(background),
        }
        logger.info(f"[SF] Layer created: bars, theme={color_theme}, mirror={mirror}, glow={glow}")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerRadial — radial spectrum visualization layer
# ---------------------------------------------------------------------------

class SF_LayerRadial:
    """Create radial/circular spectrum visualization layer descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_spectrum": (T_SF_SPECTRUM,),
                "color_theme": (["neon", "fire", "ice", "rainbow"], {
                    "default": "neon",
                }),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    def create(self, sf_spectrum, color_theme):
        colors = SF_LayerBars.COLOR_THEMES.get(color_theme, SF_LayerBars.COLOR_THEMES["neon"])
        layer = {
            "type": "radial",
            "spectrum": sf_spectrum,
            "params": {
                "colors": colors,
                "bg_color": (10, 10, 25),
            },
        }
        logger.info(f"[SF] Layer created: radial, theme={color_theme}")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerWaveform — audio waveform visualization layer
# ---------------------------------------------------------------------------

class SF_LayerWaveform:
    """Create waveform oscilloscope visualization layer descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_audio": (T_SF_AUDIO,),
                "line_color_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "line_color_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "line_color_b": ("INT", {"default": 180, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    def create(self, sf_audio, line_color_r, line_color_g, line_color_b):
        layer = {
            "type": "waveform",
            "audio": sf_audio,
            "fps": 30.0,  # will be overridden by canvas at render time
            "params": {
                "line_color": (line_color_r, line_color_g, line_color_b),
                "bg_color": (10, 10, 25),
            },
        }
        logger.info(f"[SF] Layer created: waveform")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerPianoRoll — MIDI piano roll visualization layer
# ---------------------------------------------------------------------------

class SF_LayerPianoRoll:
    """Create piano roll visualization layer descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_midi": (T_SF_MIDI,),
                "roll_window": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 30.0,
                    "tooltip": "可见时间窗口（秒）",
                }),
                "piano_height": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 0.5,
                    "tooltip": "钢琴键区高度比例",
                }),
                "smooth_keys": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用按键物理弹簧动力学平滑",
                }),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    def create(self, sf_midi, roll_window, piano_height, smooth_keys):
        layer = {
            "type": "piano_roll",
            "midi": sf_midi,
            "params": {
                "roll_window": roll_window,
                "piano_height": piano_height,
                "smooth_keys": smooth_keys,
            },
            "_key_dynamics_cache": None,  # pre-computed at render time
        }
        logger.info(f"[SF] Layer created: piano_roll, window={roll_window}s")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerGLSL — custom GLSL shader layer
# ---------------------------------------------------------------------------

class SF_LayerGLSL:
    """Create custom GLSL fragment shader layer descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fragment_shader": ("STRING", {
                    "default": """#version 330
uniform float iTime;
uniform vec2 iResolution;
uniform int iFrame;
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    vec2 uv = v_texcoord;
    vec3 col = 0.5 + 0.5*cos(iTime + uv.xyx + vec3(0,2,4));
    fragColor = vec4(col, 1.0);
}""",
                    "multiline": True,
                    "tooltip": "GLSL 片段着色器代码",
                }),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    def create(self, fragment_shader):
        layer = {
            "type": "glsl",
            "params": {
                "fragment_shader": fragment_shader,
            },
            "_glsl_renderer": None,  # initialized at render time
        }
        logger.info("[SF] Layer created: glsl")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerShaderToy — ShaderToy-compatible layer
# ---------------------------------------------------------------------------

class SF_LayerShaderToy:
    """Create ShaderToy-compatible mainImage() shader layer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shadertoy_code": ("STRING", {
                    "default": """void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float d = length(uv - 0.5);
    vec3 col = 0.5 + 0.5*cos(iTime + uv.xyx*6.283 + vec3(0,2,4));
    col *= smoothstep(0.5, 0.2, d);
    fragColor = vec4(col, 1.0);
}""",
                    "multiline": True,
                    "tooltip": "ShaderToy 兼容代码，粘贴 mainImage() 即可",
                }),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Layer"

    def create(self, shadertoy_code):
        from .shaderflow_bridge import HeadlessGLSLRenderer
        full_shader = HeadlessGLSLRenderer.wrap_shadertoy(shadertoy_code)
        layer = {
            "type": "glsl",
            "params": {
                "fragment_shader": full_shader,
            },
            "_glsl_renderer": None,
        }
        logger.info("[SF] Layer created: shadertoy")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerColorGrade — post-process color grading
# ---------------------------------------------------------------------------

class SF_LayerColorGrade:
    """Wrap another SF_LAYER with color grading post-process."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_layer": (T_SF_LAYER,),
                "sf_canvas": (T_SF_CANVAS,),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/FX"

    def create(self, sf_layer, sf_canvas, temperature, tint, exposure,
               contrast, saturation, gamma):
        layer = {
            "type": "color_grade",
            "source_layer": sf_layer,
            "canvas": sf_canvas,
            "params": {
                "temperature": temperature,
                "tint": tint,
                "exposure": exposure,
                "contrast": contrast,
                "saturation": saturation,
                "gamma": gamma,
            },
        }
        logger.info(f"[SF] Layer created: color_grade (wrapping {sf_layer.get('type', '?')})")
        return (layer,)


# ---------------------------------------------------------------------------
# SF_LayerMotionBlur — post-process motion blur
# ---------------------------------------------------------------------------

class SF_LayerMotionBlur:
    """Wrap another SF_LAYER with temporal motion blur."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_layer": (T_SF_LAYER,),
                "sf_canvas": (T_SF_CANVAS,),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "模糊强度 | 0=无 | 0.3=轻 | 0.7=重",
                }),
            },
        }

    RETURN_TYPES = (T_SF_LAYER,)
    RETURN_NAMES = ("sf_layer",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/FX"

    def create(self, sf_layer, sf_canvas, strength):
        layer = {
            "type": "motion_blur",
            "source_layer": sf_layer,
            "canvas": sf_canvas,
            "params": {"strength": strength},
            "_accum": None,
        }
        logger.info(f"[SF] Layer created: motion_blur (strength={strength})")
        return (layer,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_SF_LayerBars": SF_LayerBars,
    "VHS_SF_LayerRadial": SF_LayerRadial,
    "VHS_SF_LayerWaveform": SF_LayerWaveform,
    "VHS_SF_LayerPianoRoll": SF_LayerPianoRoll,
    "VHS_SF_LayerGLSL": SF_LayerGLSL,
    "VHS_SF_LayerShaderToy": SF_LayerShaderToy,
    "VHS_SF_LayerColorGrade": SF_LayerColorGrade,
    "VHS_SF_LayerMotionBlur": SF_LayerMotionBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SF_LayerBars": "SF Layer: Bars 📊🅥🅗🅢",
    "VHS_SF_LayerRadial": "SF Layer: Radial 🔵🅥🅗🅢",
    "VHS_SF_LayerWaveform": "SF Layer: Waveform 〰️🅥🅗🅢",
    "VHS_SF_LayerPianoRoll": "SF Layer: Piano Roll 🎹🅥🅗🅢",
    "VHS_SF_LayerGLSL": "SF Layer: Custom GLSL 🔧🅥🅗🅢",
    "VHS_SF_LayerShaderToy": "SF Layer: ShaderToy 🎮🅥🅗🅢",
    "VHS_SF_LayerColorGrade": "SF FX: Color Grade 🎨🅥🅗🅢",
    "VHS_SF_LayerMotionBlur": "SF FX: Motion Blur 💨🅥🅗🅢",
}
