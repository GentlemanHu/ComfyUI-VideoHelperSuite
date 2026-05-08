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
    PianoRollRenderer,
    HeadlessGLSLRenderer,
    read_audio_file,
    get_audio_duration,
    write_frames_to_video,
    get_available_backend,
    parse_midi_file,
)


def _start_ffmpeg_stream(output_path: str, width: int, height: int,
                         fps: float, codec: str, audio_path: str = "",
                         quality: int = 23):
    tmp_video = output_path + ".tmp.mp4" if audio_path and os.path.exists(audio_path) else output_path
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", codec, "-crf", str(quality),
        "-pix_fmt", "yuv420p",
        tmp_video,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE), tmp_video


def _finish_ffmpeg_stream(proc, tmp_video: str, output_path: str, audio_path: str = ""):
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (exit {proc.returncode})")

    if audio_path and os.path.exists(audio_path):
        mux_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_video,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ]
        subprocess.run(mux_cmd, check=True, timeout=120)
        if tmp_video != output_path:
            os.remove(tmp_video)


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
                    "default": False,
                    "tooltip": "是否同时输出帧序列（高内存）；关闭时仅输出视频和 1x1 占位 IMAGE",
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
        output_frames: bool = False,
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

        out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
        audio_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        out_path = os.path.join(out_dir, f"shaderflow_vis_{vis_mode}_{audio_hash}.{output_format}")
        codec_map = {"mp4": "libx264", "mkv": "libx264", "webm": "libvpx-vp9"}
        codec = codec_map.get(output_format, "libx264")

        # Generate frames
        pbar = ProgressBar(total_frames) if ProgressBar else None
        frames = [] if output_frames else None
        proc = tmp_video = None
        if frames is None:
            proc, tmp_video = _start_ffmpeg_stream(out_path, width, height, fps, codec, audio_path)
        dt = 1.0 / fps

        try:
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

                spectrum = spec_engine.compute(window)
                if spectrum.ndim > 1 and spectrum.shape[0] > 1:
                    spectrum_avg = spectrum.mean(axis=0)
                else:
                    spectrum_avg = spectrum.flatten()

                spec_max = spectrum_avg.max()
                if spec_max > 1e-6:
                    spectrum_avg = spectrum_avg / spec_max

                dynamics.target = spectrum_avg
                dynamics.next(dt)
                smoothed = np.clip(dynamics.value, 0, 1)

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
                    wf = window.mean(axis=0) if window.ndim > 1 else window.flatten()
                    wf_frame = renderer.render_waveform_frame(
                        wf, bg_color=(0, 0, 0),
                        line_color=theme["waveform"],
                    )
                    vis = np.clip(vis.astype(np.float32) + wf_frame.astype(np.float32) * 0.6,
                                  0, 255).astype(np.uint8)
                else:
                    vis = renderer.render_bars_frame(smoothed, bg_color=theme["bg"])

                vis = renderer.composite_with_background(vis, bg_img, bg_opacity)
                if frames is not None:
                    frames.append(vis)
                else:
                    proc.stdin.write(vis.tobytes())

                if pbar:
                    pbar.update(1)
        finally:
            if frames is None and proc is not None:
                _finish_ffmpeg_stream(proc, tmp_video, out_path, audio_path)

        if frames is not None:
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
                frames_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            else:
                frames_tensor = np.zeros((1, 1, 1, 3), dtype=np.float32)

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
# MIDI Piano Roll Node (Phase 2)
# ---------------------------------------------------------------------------

class ShaderFlowPianoRoll:
    """MIDI piano roll visualization node.
    Renders scrolling note blocks with piano keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_path": ("STRING", {
                    "default": "",
                    "tooltip": "MIDI文件路径(.mid/.midi)",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 10.0, "max": 120.0, "step": 1.0,
                }),
                "roll_window": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1,
                    "tooltip": "可视窗口时长(秒) | 决定音符滚动速度",
                }),
                "piano_height": ("FLOAT", {
                    "default": 0.12, "min": 0.05, "max": 0.4, "step": 0.01,
                    "tooltip": "钢琴键高度占比",
                }),
                "key_smooth_freq": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "琴键按压动画平滑频率",
                }),
                "key_smooth_zeta": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "琴键按压阻尼 | <1: 有弹跳效果",
                }),
            },
            "optional": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "音频文件路径(可选)。如提供则合成到视频中",
                }),
                "output_format": (["mp4", "mkv", "webm"], {
                    "default": "mp4",
                }),
                "output_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否同时输出帧序列（高内存）；关闭时仅输出视频和 1x1 占位 IMAGE",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "generate_piano_roll"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def generate_piano_roll(
        self,
        midi_path: str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        roll_window: float = 3.0,
        piano_height: float = 0.12,
        key_smooth_freq: float = 8.0,
        key_smooth_zeta: float = 0.6,
        audio_path: str = "",
        output_format: str = "mp4",
        output_frames: bool = False,
    ):
        if not midi_path or not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        notes, duration, min_note, max_note = parse_midi_file(midi_path)
        total_frames = int(math.ceil(duration * fps))
        logger.info(f"[ShaderFlow] Piano roll: {len(notes)} notes, "
                     f"{duration:.1f}s, {total_frames} frames")

        renderer = PianoRollRenderer(width=width, height=height)
        dt = 1.0 / fps

        # Per-key dynamics for press animation
        key_dynamics = [DynamicNumber(value=0.0, frequency=key_smooth_freq, zeta=key_smooth_zeta)
                        for _ in range(128)]

        pbar = ProgressBar(total_frames) if ProgressBar else None
        out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
        midi_hash = hashlib.md5(midi_path.encode()).hexdigest()[:8]
        out_path = os.path.join(out_dir, f"shaderflow_piano_{midi_hash}.{output_format}")

        codec_map = {"mp4": "libx264", "mkv": "libx264", "webm": "libvpx-vp9"}
        audio_file = audio_path if audio_path and os.path.exists(audio_path) else None
        codec = codec_map.get(output_format, "libx264")

        frames = [] if output_frames else None
        proc = tmp_video = None
        if frames is None:
            proc, tmp_video = _start_ffmpeg_stream(out_path, width, height, fps, codec, audio_file)

        try:
            for fi in range(total_frames):
                t = fi * dt

                # Update per-key dynamics
                key_vals = np.zeros(128, dtype=np.float32)
                for n in notes:
                    if n["start"] <= t <= n["end"]:
                        key_dynamics[n["note"]].target = np.array([float(n["velocity"])], dtype=np.float32)
                    else:
                        # Only decay if not currently pressed
                        if key_dynamics[n["note"]].target[0] > 0 and t > n["end"]:
                            key_dynamics[n["note"]].target = np.array([0.0], dtype=np.float32)
                for k in range(128):
                    key_dynamics[k].next(dt)
                    key_vals[k] = float(key_dynamics[k].value[0])

                frame = renderer.render_frame(
                    time=t,
                    notes=notes,
                    roll_time=roll_window,
                    min_note=min_note,
                    max_note=max_note,
                    piano_height_ratio=piano_height,
                    key_dynamics=key_vals,
                )
                if frames is not None:
                    frames.append(frame)
                else:
                    proc.stdin.write(frame.tobytes())
                if pbar:
                    pbar.update(1)
        finally:
            if frames is None and proc is not None:
                _finish_ffmpeg_stream(proc, tmp_video, out_path, audio_file)

        if frames is not None:
            write_frames_to_video(
                frames=frames,
                output_path=out_path,
                fps=fps,
                audio_path=audio_file,
                codec=codec,
            )
        logger.info(f"[ShaderFlow] Piano roll video saved: {out_path}")

        frames_tensor = None
        if output_frames and HAS_TORCH:
            arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(arr)
        if frames_tensor is None:
            frames_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32) if HAS_TORCH else np.zeros((1, 1, 1, 3), dtype=np.float32)

        return (out_path, frames_tensor)


# ---------------------------------------------------------------------------
# Custom GLSL Shader Node (Phase 3 - Route B with Route A fallback)
# ---------------------------------------------------------------------------

class ShaderFlowCustomGLSL:
    """Run custom GLSL fragment shaders headlessly.
    Route B (EGL/ModernGL) preferred, with procedural PyTorch fallback.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fragment_shader": ("STRING", {
                    "default": """#version 330
uniform float iTime;
uniform vec2 iResolution;
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    vec2 uv = v_texcoord;
    vec3 col = 0.5 + 0.5*cos(iTime + uv.xyx + vec3(0,2,4));
    fragColor = vec4(col, 1.0);
}""",
                    "multiline": True,
                    "tooltip": "GLSL Fragment Shader代码。支持iTime/iResolution/iFrame uniforms。",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                }),
                "total_frames": ("INT", {
                    "default": 120, "min": 1, "max": 100000,
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0,
                }),
            },
            "optional": {
                "output_format": (["mp4", "mkv", "webm"], {
                    "default": "mp4",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "render_glsl"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def render_glsl(
        self,
        fragment_shader: str,
        width: int = 1920,
        height: int = 1080,
        total_frames: int = 120,
        fps: float = 30.0,
        output_format: str = "mp4",
    ):
        dt = 1.0 / fps
        frames = []
        backend = get_available_backend()
        used_glsl = False

        # Route B: try headless GLSL
        if backend == "shaderflow":
            glsl = HeadlessGLSLRenderer(width=width, height=height)
            if glsl.compile(fragment_shader):
                used_glsl = True
                pbar = ProgressBar(total_frames) if ProgressBar else None
                for fi in range(total_frames):
                    t = fi * dt
                    result = glsl.render_frame(time=t, frame=fi, dt=dt)
                    if result is not None:
                        frames.append(result)
                    else:
                        used_glsl = False
                        frames = []
                        break
                    if pbar:
                        pbar.update(1)
                glsl.release()

        # Route A fallback: procedural rainbow
        if not used_glsl:
            logger.info("[ShaderFlow] GLSL not available, using PyTorch procedural fallback")
            pbar = ProgressBar(total_frames) if ProgressBar else None
            for fi in range(total_frames):
                t = fi * dt
                yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
                uv_x = xx / width
                uv_y = yy / height
                r = (0.5 + 0.5 * np.cos(t + uv_x * 6.28)) * 255
                g = (0.5 + 0.5 * np.cos(t + uv_y * 6.28 + 2.0)) * 255
                b = (0.5 + 0.5 * np.cos(t + (uv_x + uv_y) * 3.14 + 4.0)) * 255
                frame = np.stack([r, g, b], axis=-1).astype(np.uint8)
                frames.append(frame)
                if pbar:
                    pbar.update(1)

        # Write video
        out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
        sh = hashlib.md5(fragment_shader.encode()).hexdigest()[:8]
        out_path = os.path.join(out_dir, f"shaderflow_glsl_{sh}.{output_format}")
        codec_map = {"mp4": "libx264", "mkv": "libx264", "webm": "libvpx-vp9"}
        write_frames_to_video(frames=frames, output_path=out_path, fps=fps,
                              codec=codec_map.get(output_format, "libx264"))

        if HAS_TORCH:
            arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(arr)
        else:
            frames_tensor = np.stack(frames, axis=0).astype(np.float32) / 255.0

        return (out_path, frames_tensor)


# ---------------------------------------------------------------------------
# ShaderToy Compatible Node (Phase 3)
# ---------------------------------------------------------------------------

class ShaderFlowShaderToy:
    """Run ShaderToy-compatible mainImage() shaders.
    Automatically wraps ShaderToy code with proper uniforms.
    Route B (EGL) preferred, Route A (PyTorch) fallback.
    """

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
                    "tooltip": "ShaderToy兼容代码。粘贴ShaderToy的mainImage函数即可。支持iTime/iResolution/iFrame/iTimeDelta/iMouse。",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                }),
                "total_frames": ("INT", {
                    "default": 120, "min": 1, "max": 100000,
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0,
                }),
            },
            "optional": {
                "output_format": (["mp4", "mkv", "webm"], {
                    "default": "mp4",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "render_shadertoy"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def render_shadertoy(
        self,
        shadertoy_code: str,
        width: int = 1920,
        height: int = 1080,
        total_frames: int = 120,
        fps: float = 30.0,
        output_format: str = "mp4",
    ):
        # Wrap ShaderToy code into standalone fragment shader
        fragment = HeadlessGLSLRenderer.wrap_shadertoy(shadertoy_code)

        # Reuse the CustomGLSL node logic
        glsl_node = ShaderFlowCustomGLSL()
        return glsl_node.render_glsl(
            fragment_shader=fragment,
            width=width,
            height=height,
            total_frames=total_frames,
            fps=fps,
            output_format=output_format,
        )


# ---------------------------------------------------------------------------
# Motion Blur Node (Phase 3 - ShaderFX port)
# ---------------------------------------------------------------------------

class ShaderFlowMotionBlur:
    """Per-frame motion blur via frame accumulation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "输入图像序列",
                }),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "运动模糊强度 | 0=无模糊 | 0.3=轻度 | 0.7=重度",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_motion_blur"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def apply_motion_blur(self, images, strength: float = 0.3):
        if not HAS_TORCH:
            return (images,)

        imgs = images.cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
        n = imgs.shape[0]
        results = [imgs[0].copy()]
        accum = imgs[0].copy()

        for i in range(1, n):
            accum = accum * strength + imgs[i] * (1.0 - strength)
            results.append(accum.copy())

        out = np.stack(results, axis=0).astype(np.float32)
        return (torch.from_numpy(out),)


# ---------------------------------------------------------------------------
# Color Grading Node (Phase 3 - ShaderFX port)
# ---------------------------------------------------------------------------

class ShaderFlowColorGrade:
    """Cinema-style color grading."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "色温 | <0: 冷色调 | >0: 暖色调",
                }),
                "tint": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "色调 | <0: 偏绿 | >0: 偏品红",
                }),
                "exposure": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                    "tooltip": "曝光补偿 EV",
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "对比度 | 1=原始",
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "饱和度 | 0=灰度 | 1=原始",
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "伽马 | <1: 提亮 | >1: 压暗",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_color_grade"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow"

    def apply_color_grade(
        self, images, temperature=0.0, tint=0.0,
        exposure=0.0, contrast=1.0, saturation=1.0, gamma=1.0,
    ):
        if not HAS_TORCH:
            return (images,)

        imgs = images.cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
        result = imgs.copy().astype(np.float32)

        # Exposure
        if exposure != 0.0:
            result = result * (2.0 ** exposure)

        # Temperature & tint (simplified)
        if temperature != 0.0 or tint != 0.0:
            result[:, :, :, 0] = result[:, :, :, 0] + temperature * 0.1  # R
            result[:, :, :, 1] = result[:, :, :, 1] - tint * 0.05        # G
            result[:, :, :, 2] = result[:, :, :, 2] - temperature * 0.1  # B

        # Contrast
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5

        # Saturation
        if saturation != 1.0:
            luma = 0.299 * result[:, :, :, 0] + 0.587 * result[:, :, :, 1] + 0.114 * result[:, :, :, 2]
            luma = luma[:, :, :, np.newaxis]
            result = luma + (result - luma) * saturation

        # Gamma
        if gamma != 1.0:
            result = np.clip(result, 0, None)
            result = result ** (1.0 / gamma)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        return (torch.from_numpy(result),)


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_ShaderFlow_AudioVisualizer": ShaderFlowAudioVisualizer,
    "VHS_ShaderFlow_Dynamics": ShaderFlowDynamicsNode,
    "VHS_ShaderFlow_ImageEffect": ShaderFlowImageEffect,
    "VHS_ShaderFlow_PianoRoll": ShaderFlowPianoRoll,
    "VHS_ShaderFlow_CustomGLSL": ShaderFlowCustomGLSL,
    "VHS_ShaderFlow_ShaderToy": ShaderFlowShaderToy,
    "VHS_ShaderFlow_MotionBlur": ShaderFlowMotionBlur,
    "VHS_ShaderFlow_ColorGrade": ShaderFlowColorGrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_ShaderFlow_AudioVisualizer": "ShaderFlow Audio Visualizer 🎵🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_Dynamics": "ShaderFlow Dynamics (2nd Order) 🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_ImageEffect": "ShaderFlow Image Effect 🌊✨🎥🅥🅗🅢",
    "VHS_ShaderFlow_PianoRoll": "ShaderFlow Piano Roll 🎹🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_CustomGLSL": "ShaderFlow Custom GLSL 🔧🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_ShaderToy": "ShaderFlow ShaderToy 🎮🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_MotionBlur": "ShaderFlow Motion Blur 💨🌊🎥🅥🅗🅢",
    "VHS_ShaderFlow_ColorGrade": "ShaderFlow Color Grade 🎨🌊🎥🅥🅗🅢",
}
