"""
ShaderFlow Modular Nodes — Input & Data Source nodes.
SF_LoadAudio, SF_LoadMIDI, SF_CreateCanvas
"""
import os
import logging
from typing import Any, Dict

import numpy as np

from .sf_core import (
    T_SF_AUDIO, T_SF_CANVAS, T_SF_MIDI,
    logger,
)
from .shaderflow_bridge import (
    load_audio_normalized,
    parse_midi_file,
    get_audio_duration,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


# ---------------------------------------------------------------------------
# SF_LoadAudio
# ---------------------------------------------------------------------------

class SF_LoadAudio:
    """Load audio file → SF_AUDIO data.
    Supports wav, mp3, flac, ogg, m4a via ffmpeg.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": "音频文件路径 (.wav/.mp3/.flac/.ogg)",
                    "tooltip": "音频文件路径，支持所有 ffmpeg 可解码格式",
                }),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 96000, "step": 100,
                    "tooltip": "采样率 Hz",
                }),
                "channels": (["mono", "stereo"], {
                    "default": "stereo",
                    "tooltip": "声道数",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "ComfyUI原生AUDIO输入（优先于路径）",
                }),
            },
        }

    RETURN_TYPES = (T_SF_AUDIO, "FLOAT")
    RETURN_NAMES = ("sf_audio", "duration")
    FUNCTION = "load"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Input"

    def load(self, audio_path: str, sample_rate: int, channels: str, audio=None):
        ch = 1 if channels == "mono" else 2

        # Priority: ComfyUI AUDIO > file path
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            waveform = audio["waveform"]
            sr = int(audio.get("sample_rate", sample_rate))
            if HAS_TORCH and isinstance(waveform, torch.Tensor):
                samples = waveform.cpu().numpy()
            else:
                samples = np.asarray(waveform)
            # Normalize shape to (channels, N)
            if samples.ndim == 3:
                samples = samples[0]  # batch dim
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
            duration = samples.shape[1] / sr
            logger.info(f"[SF] Audio from AUDIO input: {sr}Hz, {samples.shape[0]}ch, {duration:.2f}s")
            sf_audio = {
                "samples": samples.astype(np.float32),
                "sr": sr,
                "duration": duration,
                "channels": samples.shape[0],
                "path": "",
            }
            return (sf_audio, duration)

        if not audio_path or not audio_path.strip():
            raise ValueError("请提供音频路径或连接 AUDIO 输入")

        path = audio_path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"音频文件不存在: {path}")

        sf_audio = load_audio_normalized(path, samplerate=sample_rate, channels=ch)
        return (sf_audio, sf_audio["duration"])


# ---------------------------------------------------------------------------
# SF_LoadMIDI
# ---------------------------------------------------------------------------

class SF_LoadMIDI:
    """Load MIDI file → SF_MIDI data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_path": ("STRING", {
                    "default": "",
                    "placeholder": "MIDI 文件路径 (.mid/.midi)",
                    "tooltip": "MIDI 文件路径",
                }),
            },
        }

    RETURN_TYPES = (T_SF_MIDI, "FLOAT")
    RETURN_NAMES = ("sf_midi", "duration")
    FUNCTION = "load"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Input"

    def load(self, midi_path: str):
        path = midi_path.strip()
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"MIDI 文件不存在: {path}")

        notes, duration, min_note, max_note = parse_midi_file(path)
        logger.info(
            f"[SF] MIDI loaded: {len(notes)} notes, {duration:.2f}s, "
            f"range={min_note}-{max_note}, path={path}"
        )
        sf_midi = {
            "notes": notes,
            "duration": duration,
            "min_note": min_note,
            "max_note": max_note,
            "path": path,
        }
        return (sf_midi, duration)


# ---------------------------------------------------------------------------
# SF_CreateCanvas
# ---------------------------------------------------------------------------

class SF_CreateCanvas:
    """Create canvas configuration. Intelligently infers size/duration from inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                    "tooltip": "画布宽度（像素），连接IMAGE时自动推断",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                    "tooltip": "画布高度（像素），连接IMAGE时自动推断",
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "帧率",
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                    "tooltip": "时长（秒），0=从音频/MIDI自动推断",
                }),
            },
            "optional": {
                "sf_audio": (T_SF_AUDIO, {
                    "tooltip": "连接后自动使用音频时长",
                }),
                "sf_midi": (T_SF_MIDI, {
                    "tooltip": "连接后自动使用MIDI时长",
                }),
                "image": ("IMAGE", {
                    "tooltip": "连接后自动使用图像尺寸",
                }),
            },
        }

    RETURN_TYPES = (T_SF_CANVAS,)
    RETURN_NAMES = ("sf_canvas",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Input"

    def create(self, width, height, fps, duration,
               sf_audio=None, sf_midi=None, image=None):
        w, h = int(width), int(height)

        # Auto-infer size from IMAGE
        if image is not None:
            if HAS_TORCH and isinstance(image, torch.Tensor):
                if image.ndim == 4:
                    h, w = image.shape[1], image.shape[2]
                elif image.ndim == 3:
                    h, w = image.shape[0], image.shape[1]
            logger.info(f"[SF] Canvas size from IMAGE: {w}x{h}")

        # Auto-infer duration
        dur = float(duration)
        if dur <= 0:
            if sf_audio is not None:
                dur = sf_audio.get("duration", 10.0)
                logger.info(f"[SF] Canvas duration from audio: {dur:.2f}s")
            elif sf_midi is not None:
                dur = sf_midi.get("duration", 10.0) + 1.0  # add 1s tail
                logger.info(f"[SF] Canvas duration from MIDI: {dur:.2f}s")
            else:
                dur = 10.0
                logger.info("[SF] Canvas duration default: 10.0s")

        canvas = {
            "width": w,
            "height": h,
            "fps": float(fps),
            "duration": dur,
        }
        total_frames = max(1, int(dur * fps))
        logger.info(f"[SF] Canvas created: {w}x{h} @ {fps}fps, {dur:.2f}s ({total_frames} frames)")
        return (canvas,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_SF_LoadAudio": SF_LoadAudio,
    "VHS_SF_LoadMIDI": SF_LoadMIDI,
    "VHS_SF_CreateCanvas": SF_CreateCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SF_LoadAudio": "SF Load Audio 🎵🅥🅗🅢",
    "VHS_SF_LoadMIDI": "SF Load MIDI 🎹🅥🅗🅢",
    "VHS_SF_CreateCanvas": "SF Create Canvas 🖼️🅥🅗🅢",
}
