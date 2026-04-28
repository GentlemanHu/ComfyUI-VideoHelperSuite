"""
ShaderFlow ComfyUI Nodes
Audio visualization, waveform, spectrogram nodes ported from ShaderFlow.
Backend: EGL/ShaderFlow (preferred) → PyTorch (fallback).
"""
import os
import sys
import json
import math
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger("ShaderFlowNodes")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

from .shaderflow_bridge import (
    DynamicNumber,
    SpectrogramEngine,
    TorchVisualizerRenderer,
    read_audio_file,
    get_audio_duration,
    write_frames_to_video,
    get_available_backend,
)


# ---------------------------------------------------------------------------
# Color theme presets
# ---------------------------------------------------------------------------

COLOR_THEMES = {
    "neon_cyber": {
        "bg": (8, 8, 22),
        "bars": [(0, 255, 220), (80, 120, 255), (200, 50, 255), (255, 50, 120)],
        "waveform": (0, 255, 200),
        "label": "Neon Cyber",
    },
    "sunset_glow": {
        "bg": (15, 5, 20),
        "bars": [(255, 200, 50), (255, 120, 30), (255, 50, 80), (180, 30, 120)],
        "waveform": (255, 180, 60),
        "label": "Sunset Glow",
    },
    "ocean_deep": {
        "bg": (5, 10, 25),
        "bars": [(0, 180, 255), (0, 120, 200), (30, 60, 180), (80, 40, 160)],
        "waveform": (0, 200, 255),
        "label": "Ocean Deep",
    },
    "aurora": {
        "bg": (5, 8, 18),
        "bars": [(50, 255, 100), (0, 200, 255), (120, 80, 255), (200, 50, 200)],
        "waveform": (80, 255, 150),
        "label": "Aurora",
    },
    "monochrome": {
        "bg": (0, 0, 0),
        "bars": [(200, 200, 200), (160, 160, 160), (120, 120, 120), (80, 80, 80)],
        "waveform": (220, 220, 220),
        "label": "Monochrome",
    },
    "fire": {
        "bg": (10, 2, 0),
        "bars": [(255, 255, 80), (255, 180, 0), (255, 80, 0), (180, 20, 0)],
        "waveform": (255, 160, 30),
        "label": "Fire",
    },
}


# ---------------------------------------------------------------------------
# ShaderFlow Audio Visualizer Node
# ---------------------------------------------------------------------------

class ShaderFlowAudioVisualizer:
    """Audio spectrum / waveform / radial visualization node.
    Generates video from audio input with ShaderFlow-quality rendering.
    EGL/ShaderFlow preferred, PyTorch fallback.
    """

    @classmethod
    def INPUT_TYPES(cls):
        theme_names = list(COLOR_THEMES.keys())
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "音频文件路径(mp3/wav/ogg/flac/aac)",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                    "tooltip": "输出视频宽度",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                    "tooltip": "输出视频高度",
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 10.0, "max": 120.0, "step": 1.0,
                    "tooltip": "输出视频帧率",
                }),
                "vis_mode": (["bars", "radial", "waveform", "bars+waveform"], {
                    "default": "bars",
                    "tooltip": "可视化模式 | bars: 频谱柱状图 | radial: 径向频谱 | waveform: 波形 | bars+waveform: 柱状图+波形叠加",
                }),
                "color_theme": (theme_names, {
                    "default": "neon_cyber",
                    "tooltip": "颜色主题",
                }),
                "fft_size": ("INT", {
                    "default": 12, "min": 8, "max": 15, "step": 1,
                    "tooltip": "FFT大小=2^N | 10=1024 | 12=4096(推荐) | 14=16384(高精度)",
                }),
                "spectrum_bins": ("INT", {
                    "default": 128, "min": 16, "max": 2048, "step": 8,
                    "tooltip": "频谱柱数量 | 64: 粗糙 | 128: 标准 | 256: 精细 | 512+: 超精细",
                }),
                "smoothing_freq": ("FLOAT", {
                    "default": 6.0, "min": 0.5, "max": 30.0, "step": 0.5,
                    "tooltip": "动画平滑频率(二阶系统) | 低=更平滑 | 高=更灵敏",
                }),
                "smoothing_zeta": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "阻尼比 | <1: 有弹跳 | 1: 临界阻尼 | >1: 过阻尼",
                }),
                "mirror_bars": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "柱状图上下镜像",
                }),
                "glow_effect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "发光效果",
                }),
                "min_frequency": ("FLOAT", {
                    "default": 20.0, "min": 10.0, "max": 1000.0, "step": 10.0,
                    "tooltip": "最低频率 Hz",
                }),
                "max_frequency": ("FLOAT", {
                    "default": 16000.0, "min": 1000.0, "max": 22050.0, "step": 100.0,
                    "tooltip": "最高频率 Hz",
                }),
            },
            "optional": {
                "background_image": ("IMAGE", {
                    "tooltip": "背景图像(可选)，会被暗化后作为背景",
                }),
                "bg_opacity": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "背景图不透明度 | 0=纯黑 | 1=全亮",
                }),
                "output_format": (["mp4", "mkv", "webm"], {
                    "default": "mp4",
                }),
                "output_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否同时输出帧序列",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "generate_visualization"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def generate_visualization(
        self,
        audio_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        vis_mode: str = "bars",
        color_theme: str = "neon_cyber",
        fft_size: int = 12,
        spectrum_bins: int = 128,
        smoothing_freq: float = 6.0,
        smoothing_zeta: float = 0.8,
        mirror_bars: bool = True,
        glow_effect: bool = True,
        min_frequency: float = 20.0,
        max_frequency: float = 16000.0,
        background_image=None,
        bg_opacity: float = 0.25,
        output_format: str = "mp4",
        output_frames: bool = True,
    ):
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        theme = COLOR_THEMES.get(color_theme, COLOR_THEMES["neon_cyber"])
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            raise ValueError(f"Cannot determine audio duration: {audio_path}")

        total_frames = int(math.ceil(duration * fps))
        logger.info(f"[ShaderFlow] Generating {vis_mode} visualization: "
                     f"{total_frames} frames @ {fps}fps, {width}x{height}")

        # Read audio
        samplerate = 44100
        audio_data, samplerate = read_audio_file(audio_path, samplerate=samplerate)
        # audio_data shape: (channels, total_samples)
        channels = audio_data.shape[0]

        # Build spectrogram engine
        spec_engine = SpectrogramEngine(
            samplerate=samplerate,
            fft_n=fft_size,
            min_freq=min_frequency,
            max_freq=max_frequency,
            bins=spectrum_bins,
            channels=channels,
        )

        # Dynamics for smooth animation
        dynamics = DynamicNumber(
            value=np.zeros(spectrum_bins, dtype=np.float32),
            frequency=smoothing_freq,
            zeta=smoothing_zeta,
        )

        # Renderer
        renderer = TorchVisualizerRenderer(width=width, height=height)

        # Background image processing
        bg_img = None
        if background_image is not None:
            if HAS_TORCH and isinstance(background_image, torch.Tensor):
                # ComfyUI IMAGE tensor: (B, H, W, C), float 0-1
                bg_np = (background_image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                bg_np = np.asarray(background_image, dtype=np.uint8)
            bg_img = bg_np

        # Generate frames
        pbar = ProgressBar(total_frames) if ProgressBar else None
        frames = []
        dt = 1.0 / fps

        for fi in range(total_frames):
            t = fi / fps
            sample_center = int(t * samplerate)
            half_window = spec_engine.fft_size // 2

            # Extract audio window
            start = max(0, sample_center - half_window)
            end = start + spec_engine.fft_size
            if end > audio_data.shape[1]:
                end = audio_data.shape[1]
                start = max(0, end - spec_engine.fft_size)

            window = audio_data[:, start:end]
            if window.shape[1] < spec_engine.fft_size:
                pad = spec_engine.fft_size - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad)))

            # Compute spectrum
            spectrum = spec_engine.compute(window)
            # Average channels
            if spectrum.ndim > 1 and spectrum.shape[0] > 1:
                spectrum_avg = spectrum.mean(axis=0)
            else:
                spectrum_avg = spectrum.flatten()

            # Normalize
            spec_max = spectrum_avg.max()
            if spec_max > 1e-6:
                spectrum_avg = spectrum_avg / spec_max

            # Apply dynamics smoothing
            dynamics.target = spectrum_avg
            dynamics.next(dt)
            smoothed = np.clip(dynamics.value, 0, 1)

            # Render frame based on mode
            if vis_mode == "bars":
                vis = renderer.render_bars_frame(
                    smoothed, bg_color=theme["bg"],
                    bar_colors=theme["bars"],
                    glow=glow_effect, mirror=mirror_bars,
                )
            elif vis_mode == "radial":
                vis = renderer.render_radial_frame(
                    smoothed, bg_color=theme["bg"],
                    bar_colors=theme["bars"],
                )
            elif vis_mode == "waveform":
                # Get raw waveform samples for this frame
                wf = window.mean(axis=0) if window.ndim > 1 else window.flatten()
                vis = renderer.render_waveform_frame(
                    wf, bg_color=theme["bg"],
                    line_color=theme["waveform"],
                )
            elif vis_mode == "bars+waveform":
                vis = renderer.render_bars_frame(
                    smoothed, bg_color=theme["bg"],
                    bar_colors=theme["bars"],
                    glow=glow_effect, mirror=mirror_bars,
                )
                # Overlay waveform
                wf = window.mean(axis=0) if window.ndim > 1 else window.flatten()
                wf_frame = renderer.render_waveform_frame(
                    wf, bg_color=(0, 0, 0),
                    line_color=theme["waveform"],
                )
                vis = np.clip(vis.astype(np.float32) + wf_frame.astype(np.float32) * 0.6,
                              0, 255).astype(np.uint8)
            else:
                vis = renderer.render_bars_frame(smoothed, bg_color=theme["bg"])

            # Composite background
            vis = renderer.composite_with_background(vis, bg_img, bg_opacity)
            frames.append(vis)

            if pbar:
                pbar.update(1)

        # Write video
        out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
        audio_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        out_path = os.path.join(out_dir, f"shaderflow_vis_{vis_mode}_{audio_hash}.{output_format}")

        codec_map = {"mp4": "libx264", "mkv": "libx264", "webm": "libvpx-vp9"}
        codec = codec_map.get(output_format, "libx264")

        write_frames_to_video(
            frames=frames,
            output_path=out_path,
            fps=fps,
            audio_path=audio_path,
            codec=codec,
        )
        logger.info(f"[ShaderFlow] Video saved: {out_path}")

        # Convert frames to ComfyUI IMAGE tensor
        frames_tensor = None
        if output_frames and HAS_TORCH:
            # Stack to (N, H, W, 3) float32 0-1
            arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(arr)
        elif output_frames:
            frames_tensor = np.stack(frames, axis=0).astype(np.float32) / 255.0

        if frames_tensor is None:
            if HAS_TORCH:
                frames_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32)
            else:
                frames_tensor = np.zeros((1, height, width, 3), dtype=np.float32)

        return (out_path, frames_tensor)


# ---------------------------------------------------------------------------
# ShaderFlow Dynamics Node
# ---------------------------------------------------------------------------

class ShaderFlowDynamicsNode:
    """Second-order dynamics system from ShaderFlow.
    Animates a float value with spring-damper physics.
    Outputs per-frame animated values for driving other nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 120, "min": 1, "max": 100000,
                    "tooltip": "总帧数",
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0,
                    "tooltip": "帧率",
                }),
                "wave_type": (["sine", "square", "triangle", "sawtooth", "pulse", "constant"], {
                    "default": "sine",
                    "tooltip": "目标值波形 | sine: 正弦 | square: 方波 | triangle: 三角 | sawtooth: 锯齿",
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "波形频率 Hz",
                }),
                "amplitude": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "波形振幅",
                }),
                "offset": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "波形偏移",
                }),
                "frequency": ("FLOAT", {
                    "default": 4.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "二阶系统自然频率 | 越高跟踪越快",
                }),
                "zeta": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 3.0, "step": 0.05,
                    "tooltip": "阻尼比 | <1: 弹跳 | 1: 临界阻尼 | >1: 过阻尼(慢)",
                }),
                "response": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "初始响应 | 0: 无超调 | >0: 预测性 | <0: 滞后",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("values_json",)
    FUNCTION = "compute_dynamics"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def compute_dynamics(
        self,
        total_frames: int,
        fps: float,
        wave_type: str,
        wave_frequency: float,
        amplitude: float,
        offset: float,
        frequency: float,
        zeta: float,
        response: float,
    ):
        dt = 1.0 / fps
        dyn = DynamicNumber(
            value=offset,
            frequency=frequency,
            zeta=zeta,
            response=response,
        )

        values = []
        for fi in range(total_frames):
            t = fi * dt
            phase = 2 * math.pi * wave_frequency * t

            if wave_type == "sine":
                target = amplitude * math.sin(phase) + offset
            elif wave_type == "square":
                target = amplitude * (1.0 if math.sin(phase) >= 0 else -1.0) + offset
            elif wave_type == "triangle":
                target = amplitude * (2.0 * abs(2.0 * (t * wave_frequency - math.floor(t * wave_frequency + 0.5))) - 1.0) + offset
            elif wave_type == "sawtooth":
                target = amplitude * (2.0 * (t * wave_frequency - math.floor(t * wave_frequency)) - 1.0) + offset
            elif wave_type == "pulse":
                target = amplitude * (1.0 if (t * wave_frequency) % 1.0 < 0.1 else 0.0) + offset
            else:  # constant
                target = amplitude + offset

            dyn.target = np.array([target], dtype=np.float32)
            dyn.next(dt)
            values.append(float(dyn.value[0]))

        return (json.dumps(values),)


# ---------------------------------------------------------------------------
# ShaderFlow Image Effect Node (simple shader-like effects via numpy)
# ---------------------------------------------------------------------------

class ShaderFlowImageEffect:
    """Apply shader-like visual effects to image sequences.
    Zoom, rotate, color shift, vignette — all implemented in numpy/torch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "输入图像序列 (B, H, W, C)",
                }),
                "effect": (["zoom_pulse", "rotate", "color_shift", "vignette", "chromatic_aberration", "none"], {
                    "default": "zoom_pulse",
                    "tooltip": "效果类型",
                }),
                "intensity": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "效果强度",
                }),
                "frequency": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 5.0, "step": 0.01,
                    "tooltip": "动画频率 Hz",
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0,
                }),
                "smooth_freq": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 20.0,
                    "tooltip": "二阶系统平滑频率",
                }),
                "smooth_zeta": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0,
                    "tooltip": "二阶系统阻尼",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_effect"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def apply_effect(
        self,
        images,
        effect: str = "zoom_pulse",
        intensity: float = 0.1,
        frequency: float = 0.5,
        fps: float = 30.0,
        smooth_freq: float = 4.0,
        smooth_zeta: float = 1.0,
    ):
        if not HAS_TORCH or not HAS_CV2:
            return (images,)

        imgs = images.cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
        n, h, w, c = imgs.shape
        dt = 1.0 / fps

        dyn = DynamicNumber(value=0.0, frequency=smooth_freq, zeta=smooth_zeta)
        results = []

        for fi in range(n):
            t = fi * dt
            target = intensity * math.sin(2 * math.pi * frequency * t)
            dyn.target = np.array([target], dtype=np.float32)
            dyn.next(dt)
            val = float(dyn.value[0])

            frame = (imgs[fi] * 255).astype(np.uint8) if imgs[fi].max() <= 1.0 else imgs[fi].astype(np.uint8)

            if effect == "zoom_pulse":
                scale = 1.0 + val
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            elif effect == "rotate":
                angle_deg = val * 30  # max 30 degrees
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            elif effect == "color_shift":
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + val * 60) % 180
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            elif effect == "vignette":
                vy, vx = np.mgrid[0:h, 0:w].astype(np.float32)
                vx = (vx / w - 0.5) * 2
                vy = (vy / h - 0.5) * 2
                dist = np.sqrt(vx ** 2 + vy ** 2)
                vig = 1.0 - np.clip(dist * (0.5 + abs(val)), 0, 1)
                vig = vig[:, :, np.newaxis]
                frame = (frame.astype(np.float32) * vig).astype(np.uint8)
            elif effect == "chromatic_aberration":
                shift = max(1, int(abs(val) * 10))
                result = frame.copy()
                if frame.shape[2] >= 3:
                    result[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1)
                    result[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)
                frame = result

            results.append(frame.astype(np.float32) / 255.0)

        out = np.stack(results, axis=0)
        return (torch.from_numpy(out),)


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_ShaderFlow_AudioVisualizer": ShaderFlowAudioVisualizer,
    "VHS_ShaderFlow_Dynamics": ShaderFlowDynamicsNode,
    "VHS_ShaderFlow_ImageEffect": ShaderFlowImageEffect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_ShaderFlow_AudioVisualizer": "ShaderFlow Audio Visualizer 🎵🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_Dynamics": "ShaderFlow Dynamics (2nd Order) 🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_ImageEffect": "ShaderFlow Image Effect 🌊✨🎥🅥🅗🅢",
}
