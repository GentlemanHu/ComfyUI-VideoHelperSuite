"""
ShaderFlow Pipeline Nodes — Streamlined one-wire chain.

Pipeline = single SF_PIPELINE object carrying canvas + audio + midi + spectrum + layers.
Each node takes pipeline in, enriches it, outputs pipeline.

Workflow: SF_Pipeline → SF_AddSpectrum → SF_AddVisualizer → SF_Render
(4 nodes, 3 wires — vs atomic nodes which need 7+ nodes and many wires)

Atomic nodes (sf_nodes_input/analysis/layer/output) are kept for advanced fine control.
"""
import os
import math
import logging
import copy
from typing import Any, Dict, List, Optional

import numpy as np

from .sf_core import (
    T_SF_AUDIO, T_SF_SPECTRUM, T_SF_CURVE, T_SF_MIDI,
    T_SF_CANVAS, T_SF_LAYER, T_SF_PIPELINE,
    render_layer_frame, get_total_frames,
    logger,
)
from .shaderflow_bridge import (
    load_audio_normalized,
    compute_spectrum_batch,
    parse_midi_file,
    get_audio_duration,
    DynamicNumber,
    SpectrogramEngine,
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


def _make_pipeline(canvas, audio=None, midi=None, spectrum=None,
                   background=None, layers=None):
    """Create a new SF_PIPELINE dict."""
    return {
        "canvas": canvas,
        "audio": audio,
        "midi": midi,
        "spectrum": spectrum,
        "background": background,
        "layers": layers or [],
    }


def _clone_pipeline(pipe):
    """Shallow-clone pipeline so upstream nodes aren't mutated."""
    clone = {
        "canvas": dict(pipe["canvas"]),
        "audio": pipe.get("audio"),
        "midi": pipe.get("midi"),
        "spectrum": pipe.get("spectrum"),
        "background": pipe.get("background"),
        "layers": list(pipe.get("layers", [])),
    }
    if "audio_rms" in pipe:
        clone["audio_rms"] = pipe["audio_rms"]
    return clone


def _composite_all_layers(layers, frame_idx, canvas, w, h):
    """Render all layers and composite them together.

    Smart compositing rules:
    - DepthFlow layers render the full parallax scene as the base.
    - When DepthFlow exists, visualizer layers render on BLACK background
      and are additively overlaid so only bright spectrum elements show.
    - Post-process layers (color_grade/motion_blur) apply to the final result.
    """
    if not layers:
        return np.zeros((h, w, 3), dtype=np.uint8)

    visual_types = ("bars", "radial", "waveform", "piano_roll", "glsl")
    base_types = ("depthflow",)
    fx_types = ("color_grade", "motion_blur")

    has_base = any(l.get("type") in base_types for l in layers)
    base_frame = None
    post_fx_layers = []

    for layer in layers:
        ltype = layer.get("type", "")

        if ltype in fx_types:
            post_fx_layers.append(layer)
            continue

        if ltype in base_types:
            rendered = render_layer_frame(layer, frame_idx, canvas)
            if rendered.shape[0] != h or rendered.shape[1] != w:
                try:
                    import cv2
                    rendered = cv2.resize(rendered, (w, h))
                except ImportError:
                    pass
            base_frame = rendered

        elif ltype in visual_types:
            # When DepthFlow exists, force visualizer onto BLACK background
            # so only the bright bars/spectrum overlay, not a second copy of the image
            if has_base:
                saved_bg = layer.get("background_frame")
                layer["background_frame"] = None
                rendered = render_layer_frame(layer, frame_idx, canvas)
                layer["background_frame"] = saved_bg
            else:
                rendered = render_layer_frame(layer, frame_idx, canvas)

            if rendered.shape[0] != h or rendered.shape[1] != w:
                try:
                    import cv2
                    rendered = cv2.resize(rendered, (w, h))
                except ImportError:
                    pass

            if base_frame is None:
                base_frame = rendered
            else:
                # Additive blend: bright visualizer pixels glow on top of base
                base_f = base_frame.astype(np.float32)
                over_f = rendered.astype(np.float32)
                base_frame = np.clip(base_f + over_f * 0.7, 0, 255).astype(np.uint8)
        else:
            rendered = render_layer_frame(layer, frame_idx, canvas)
            if rendered.shape[0] != h or rendered.shape[1] != w:
                try:
                    import cv2
                    rendered = cv2.resize(rendered, (w, h))
                except ImportError:
                    pass
            base_frame = rendered if base_frame is None else rendered

    if base_frame is None:
        base_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply post-process FX on the composited result
    for fx_layer in post_fx_layers:
        params = fx_layer.get("params", {})
        if fx_layer["type"] == "color_grade":
            img = base_frame.astype(np.float32) / 255.0
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
            base_frame = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif fx_layer["type"] == "motion_blur":
            strength = params.get("strength", 0.3)
            accum = fx_layer.get("_accum")
            if accum is None or accum.shape != base_frame.shape:
                fx_layer["_accum"] = base_frame.astype(np.float32)
            else:
                blended = accum * strength + base_frame.astype(np.float32) * (1.0 - strength)
                fx_layer["_accum"] = blended
                base_frame = np.clip(blended, 0, 255).astype(np.uint8)

    return base_frame


# ---------------------------------------------------------------------------
# SF_Pipeline — create pipeline (entry point)
# ---------------------------------------------------------------------------

class SF_Pipeline:
    """Create a ShaderFlow pipeline. One node to set up everything.
    Auto-infers canvas size from IMAGE, duration from audio/MIDI.
    Audio, MIDI, and background image are stored in the pipeline
    and automatically available to all downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 8,
                    "tooltip": "画布宽度（连接IMAGE时自动推断）",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 8,
                    "tooltip": "画布高度（连接IMAGE时自动推断）",
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0,
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                    "tooltip": "时长（秒），0=从音频/MIDI自动推断",
                }),
            },
            "optional": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "音频文件路径，自动加载并推断时长",
                }),
                "midi_path": ("STRING", {
                    "default": "",
                    "tooltip": "MIDI文件路径",
                }),
                "image": ("IMAGE", {
                    "tooltip": "背景图像，同时自动推断画布尺寸",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "ComfyUI原生AUDIO输入（优先于路径）",
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def create(self, width, height, fps, duration,
               audio_path="", midi_path="", image=None, audio=None):
        w, h = int(width), int(height)

        # --- Background image & auto size ---
        bg = None
        if image is not None:
            if HAS_TORCH and isinstance(image, torch.Tensor):
                bg = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                bg = (np.asarray(image) * 255).astype(np.uint8)
            if bg.ndim == 3:
                bg = bg[np.newaxis, :]
            h, w = bg.shape[1], bg.shape[2]
            logger.info(f"[SF Pipeline] Image → canvas {w}x{h}, stored as background")

        # --- Audio ---
        sf_audio = None
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            waveform = audio["waveform"]
            sr = int(audio.get("sample_rate", 44100))
            if HAS_TORCH and isinstance(waveform, torch.Tensor):
                samples = waveform.cpu().numpy()
            else:
                samples = np.asarray(waveform)
            if samples.ndim == 3:
                samples = samples[0]
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
            sf_audio = {
                "samples": samples.astype(np.float32),
                "sr": sr,
                "duration": samples.shape[1] / sr,
                "channels": samples.shape[0],
                "path": "",
            }
            logger.info(f"[SF Pipeline] AUDIO input: {sr}Hz, {sf_audio['duration']:.2f}s")
        elif audio_path and audio_path.strip() and os.path.exists(audio_path.strip()):
            sf_audio = load_audio_normalized(audio_path.strip())

        # --- MIDI ---
        sf_midi = None
        if midi_path and midi_path.strip() and os.path.exists(midi_path.strip()):
            notes, midi_dur, min_n, max_n = parse_midi_file(midi_path.strip())
            sf_midi = {
                "notes": notes, "duration": midi_dur,
                "min_note": min_n, "max_note": max_n,
                "path": midi_path.strip(),
            }
            logger.info(f"[SF Pipeline] MIDI: {len(notes)} notes, {midi_dur:.2f}s")

        # --- Duration auto-infer ---
        dur = float(duration)
        if dur <= 0:
            if sf_audio:
                dur = sf_audio["duration"]
            elif sf_midi:
                dur = sf_midi["duration"] + 1.0
            else:
                dur = 10.0
            logger.info(f"[SF Pipeline] Duration auto: {dur:.2f}s")

        canvas = {"width": w, "height": h, "fps": float(fps), "duration": dur}
        total = max(1, int(dur * fps))
        logger.info(f"[SF Pipeline] Created: {w}x{h}@{fps}fps, {dur:.2f}s ({total}f)")

        pipe = _make_pipeline(
            canvas=canvas,
            audio=sf_audio,
            midi=sf_midi,
            background=bg,
        )
        return (pipe,)


# ---------------------------------------------------------------------------
# SF_AddSpectrum — compute spectrum and store in pipeline
# ---------------------------------------------------------------------------

class SF_AddSpectrum:
    """Compute FFT spectrum from pipeline audio. Stores result in pipeline.
    Downstream visualizer nodes automatically use it — no extra wiring needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "fft_power": ("INT", {
                    "default": 12, "min": 8, "max": 16,
                    "tooltip": "FFT窗口 = 2^N (12=4096)",
                }),
                "spectrum_bins": ("INT", {
                    "default": 128, "min": 16, "max": 1024, "step": 16,
                }),
                "min_freq": ("FLOAT", {"default": 20.0, "min": 10.0, "max": 2000.0}),
                "max_freq": ("FLOAT", {"default": 16000.0, "min": 1000.0, "max": 22000.0}),
                "smooth_freq": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "平滑频率（0=不平滑）",
                }),
                "smooth_zeta": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 3.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "compute"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def compute(self, pipeline, fft_power, spectrum_bins,
                min_freq, max_freq, smooth_freq, smooth_zeta):
        pipe = _clone_pipeline(pipeline)

        if pipe["audio"] is None:
            logger.warning("[SF Pipeline] No audio in pipeline, skipping spectrum")
            return (pipe,)

        fps = pipe["canvas"]["fps"]
        pbar = ProgressBar(100) if ProgressBar else None

        def pcb(cur, tot):
            if pbar and tot > 0:
                pbar.update_absolute(int(cur / tot * 80))

        spectrum = compute_spectrum_batch(
            audio_data=pipe["audio"], fps=fps, fft_n=fft_power,
            bins=spectrum_bins, min_freq=min_freq, max_freq=max_freq,
            progress_cb=pcb,
        )

        # Apply smoothing if requested
        if smooth_freq > 0:
            bins_data = spectrum["bins"]
            dt = 1.0 / fps
            n_frames, n_bins = bins_data.shape
            logger.info(f"[SF Pipeline] Smoothing: freq={smooth_freq}, zeta={smooth_zeta}")
            dynamics = [DynamicNumber(0.0, frequency=smooth_freq, zeta=smooth_zeta)
                        for _ in range(n_bins)]
            for fi in range(n_frames):
                for bi in range(n_bins):
                    dynamics[bi].target = bins_data[fi, bi]
                    bins_data[fi, bi] = max(0.0, float(dynamics[bi].next(dt)))
            spectrum["bins"] = bins_data

        if pbar:
            pbar.update_absolute(90)

        # Compute per-frame RMS for audio-reactive DepthFlow
        audio_data = pipe["audio"]
        samples = audio_data["samples"]
        sr = audio_data["sr"]
        total_frames = max(1, int(pipe["canvas"]["duration"] * fps))
        rms_arr = np.zeros(total_frames, dtype=np.float32)
        hop = max(1, int(sr / fps))
        mono = samples[0] if samples.ndim == 2 else samples
        for fi in range(total_frames):
            start = fi * hop
            end = min(start + hop, len(mono))
            if start < len(mono):
                chunk = mono[start:end]
                rms_arr[fi] = float(np.sqrt(np.mean(chunk ** 2)))
        # Normalize to 0..1
        rms_max = rms_arr.max()
        if rms_max > 0:
            rms_arr = rms_arr / rms_max
        pipe["audio_rms"] = rms_arr
        logger.info(f"[SF Pipeline] Computed audio_rms: {total_frames} frames, peak={rms_max:.4f}")

        if pbar:
            pbar.update_absolute(100)

        pipe["spectrum"] = spectrum
        return (pipe,)


# ---------------------------------------------------------------------------
# SF_AddVisualizer — add visualization layer to pipeline
# ---------------------------------------------------------------------------

class SF_AddVisualizer:
    """Add a visualization layer to the pipeline.
    Automatically uses spectrum/audio/midi/background from the pipeline.
    """

    COLOR_THEMES = {
        "neon": [(0, 255, 220), (120, 80, 255), (255, 50, 180), (255, 160, 40)],
        "fire": [(255, 50, 0), (255, 150, 0), (255, 220, 50), (255, 255, 180)],
        "ice": [(0, 100, 255), (0, 200, 255), (150, 230, 255), (220, 245, 255)],
        "rainbow": [(255, 0, 0), (255, 165, 0), (255, 255, 0),
                     (0, 255, 0), (0, 0, 255), (128, 0, 128)],
        "mono_cyan": [(0, 80, 100), (0, 200, 220), (0, 255, 255)],
        "mono_magenta": [(80, 0, 60), (200, 0, 150), (255, 50, 200)],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "vis_type": (["bars", "radial", "waveform", "piano_roll"], {
                    "default": "bars",
                    "tooltip": "可视化类型",
                }),
                "color_theme": (["neon", "fire", "ice", "rainbow",
                                 "mono_cyan", "mono_magenta"], {
                    "default": "neon",
                }),
                "mirror": ("BOOLEAN", {"default": True}),
                "glow": ("BOOLEAN", {"default": True}),
                "use_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用管道中的背景图像",
                }),
            },
            "optional": {
                "roll_window": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 30.0,
                    "tooltip": "钢琴卷帘可见时间窗口（秒）",
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def add(self, pipeline, vis_type, color_theme, mirror, glow,
            use_background, roll_window=3.0):
        pipe = _clone_pipeline(pipeline)
        colors = self.COLOR_THEMES.get(color_theme, self.COLOR_THEMES["neon"])

        # Build background frame getter from pipeline
        bg_getter = None
        if use_background and pipe["background"] is not None:
            bg_arr = pipe["background"]  # (N, H, W, 3)
            n_bg = bg_arr.shape[0]

            def bg_getter(frame_idx, w, h):
                idx = min(frame_idx, n_bg - 1)
                frame = bg_arr[idx]
                if frame.shape[0] != h or frame.shape[1] != w:
                    try:
                        import cv2
                        return cv2.resize(frame, (w, h))
                    except ImportError:
                        pass
                return frame

        if vis_type == "bars":
            if pipe["spectrum"] is None:
                logger.warning("[SF Pipeline] Bars需要频谱，请先连接 SF Add Spectrum")
            layer = {
                "type": "bars",
                "spectrum": pipe["spectrum"],
                "params": {"mirror": mirror, "glow": glow,
                           "colors": colors, "bg_color": (10, 10, 25)},
                "background_frame": bg_getter,
            }

        elif vis_type == "radial":
            if pipe["spectrum"] is None:
                logger.warning("[SF Pipeline] Radial需要频谱，请先连接 SF Add Spectrum")
            layer = {
                "type": "radial",
                "spectrum": pipe["spectrum"],
                "params": {"colors": colors, "bg_color": (10, 10, 25)},
                "background_frame": bg_getter,
            }

        elif vis_type == "waveform":
            if pipe["audio"] is None:
                logger.warning("[SF Pipeline] Waveform需要音频")
            layer = {
                "type": "waveform",
                "audio": pipe["audio"],
                "fps": pipe["canvas"]["fps"],
                "params": {"line_color": colors[0] if colors else (0, 255, 180),
                           "bg_color": (10, 10, 25)},
                "background_frame": bg_getter,
            }

        elif vis_type == "piano_roll":
            if pipe["midi"] is None:
                logger.warning("[SF Pipeline] PianoRoll需要MIDI")
            layer = {
                "type": "piano_roll",
                "midi": pipe["midi"],
                "params": {"roll_window": roll_window,
                           "piano_height": 0.15,
                           "smooth_keys": True},
                "_key_dynamics_cache": None,
            }
        else:
            layer = {"type": "bars", "spectrum": pipe["spectrum"],
                     "params": {"mirror": True, "glow": True,
                                "colors": colors, "bg_color": (10, 10, 25)}}

        pipe["layers"].append(layer)
        logger.info(f"[SF Pipeline] Added visualizer: {vis_type}, theme={color_theme}")
        return (pipe,)


# ---------------------------------------------------------------------------
# SF_AddEffect — add post-processing effect to pipeline
# ---------------------------------------------------------------------------

class SF_AddEffect:
    """Add a post-processing effect wrapping the last layer in the pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "effect": (["color_grade", "motion_blur"], {
                    "default": "color_grade",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "blur_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "运动模糊强度",
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def add(self, pipeline, effect,
            temperature=0.0, tint=0.0, exposure=0.0,
            contrast=1.0, saturation=1.0, gamma=1.0,
            blur_strength=0.3):
        pipe = _clone_pipeline(pipeline)

        if not pipe["layers"]:
            logger.warning("[SF Pipeline] No layer to apply effect on")
            return (pipe,)

        # Wrap the last layer with the effect
        source_layer = pipe["layers"][-1]
        canvas = pipe["canvas"]

        if effect == "color_grade":
            fx_layer = {
                "type": "color_grade",
                "source_layer": source_layer,
                "canvas": canvas,
                "params": {
                    "temperature": temperature, "tint": tint,
                    "exposure": exposure, "contrast": contrast,
                    "saturation": saturation, "gamma": gamma,
                },
            }
        elif effect == "motion_blur":
            fx_layer = {
                "type": "motion_blur",
                "source_layer": source_layer,
                "canvas": canvas,
                "params": {"strength": blur_strength},
                "_accum": None,
            }
        else:
            return (pipe,)

        # Replace last layer with wrapped version
        pipe["layers"][-1] = fx_layer
        logger.info(f"[SF Pipeline] Added effect: {effect}")
        return (pipe,)


# ---------------------------------------------------------------------------
# SF_PipelineRender — render pipeline to video
# ---------------------------------------------------------------------------

class SF_PipelineRender:
    """Render the pipeline to video. Streams frames to ffmpeg, O(1) memory.
    Uses the LAST layer in the pipeline as the render source.
    Audio from pipeline is automatically muxed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "output_prefix": ("STRING", {"default": "sf_pipe_"}),
                "output_format": (["mp4", "mkv", "webm"], {"default": "mp4"}),
                "codec": (["libx264", "libx265", "libvpx-vp9"], {"default": "libx264"}),
                "quality": ("INT", {"default": 23, "min": 0, "max": 51}),
                "output_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否同时输出IMAGE tensor（⚠️高内存）",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def render(self, pipeline, output_prefix, output_format,
               codec, quality, output_frames):
        import subprocess
        import hashlib
        import tempfile

        pipe = pipeline
        canvas = dict(pipe["canvas"])
        w, h = canvas["width"], canvas["height"]
        fps = canvas["fps"]
        total = get_total_frames(canvas)

        if not pipe["layers"]:
            raise ValueError("Pipeline has no layers. Add a visualizer first.")

        # Inject audio_rms into canvas for audio-reactive layers
        if "audio_rms" in pipe:
            canvas["audio_rms"] = pipe["audio_rms"]

        all_layers = pipe["layers"]
        layer_types = [l.get('type', '?') for l in all_layers]
        logger.info(f"[SF Pipeline] ═══ Render Start ═══")
        logger.info(f"[SF Pipeline] {w}x{h}@{fps}fps, {total}f, layers={layer_types}")

        # Prepare ALL layers (GLSL init, key dynamics, etc.)
        from .sf_nodes_output import _prepare_layer_for_render, _cleanup_layer
        for layer in all_layers:
            _prepare_layer_for_render(layer, canvas)

        # Resolve audio
        audio_wav = None
        if pipe["audio"] is not None:
            import wave
            samples = pipe["audio"]["samples"]
            sr = pipe["audio"]["sr"]
            int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            if int16.ndim == 2:
                int16 = int16.T
            try:
                import folder_paths
                out_dir = folder_paths.get_output_directory()
            except ImportError:
                out_dir = tempfile.gettempdir()
            audio_wav = os.path.join(out_dir, f"sf_pipe_audio_{id(pipe) % 99999}.wav")
            with wave.open(audio_wav, "w") as wf:
                wf.setnchannels(pipe["audio"].get("channels", 2))
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(int16.tobytes())

        # Output path
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except ImportError:
            out_dir = tempfile.gettempdir()
        lhash = hashlib.md5(str(layer_types).encode()).hexdigest()[:6]
        out_path = os.path.join(out_dir, f"{output_prefix}{lhash}.{output_format}")
        tmp_video = out_path + ".tmp.mp4" if audio_wav else out_path

        # ffmpeg pipe
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-",
            "-c:v", codec, "-crf", str(quality),
            "-pix_fmt", "yuv420p", tmp_video,
        ]

        pbar = ProgressBar(100) if ProgressBar else None
        collected = [] if output_frames else None

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fi in range(total):
                frame = _composite_all_layers(all_layers, fi, canvas, w, h)
                proc.stdin.write(frame.tobytes())
                if collected is not None:
                    collected.append(frame)
                if pbar:
                    pbar.update_absolute(int(fi / total * 85))
                if fi > 0 and fi % max(1, total // 10) == 0:
                    logger.info(f"[SF Pipeline] Frame {fi}/{total} ({fi/total*100:.0f}%)")
        finally:
            proc.stdin.close()

        proc.wait()
        if pbar:
            pbar.update_absolute(90)
        if proc.returncode != 0:
            for layer in all_layers:
                _cleanup_layer(layer)
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")

        # Mux audio
        if audio_wav:
            mux = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", tmp_video, "-i", audio_wav,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", out_path,
            ]
            subprocess.run(mux, check=True, timeout=300)
            try:
                os.remove(tmp_video)
                os.remove(audio_wav)
            except OSError:
                pass

        if pbar:
            pbar.update_absolute(100)
        for layer in all_layers:
            _cleanup_layer(layer)

        sz = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"[SF Pipeline] ═══ Done: {out_path} ({sz:.1f}MB) ═══")

        frames_tensor = None
        if collected:
            arr = np.stack(collected, axis=0).astype(np.float32) / 255.0
            if HAS_TORCH:
                frames_tensor = torch.from_numpy(arr)
            else:
                frames_tensor = arr

        return (out_path, frames_tensor)


# ---------------------------------------------------------------------------
# SF_AddGLSL — add custom GLSL/ShaderToy layer to pipeline
# ---------------------------------------------------------------------------

class SF_AddGLSL:
    """Add custom GLSL or ShaderToy shader layer to pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "shader_type": (["glsl", "shadertoy"], {"default": "shadertoy"}),
                "code": ("STRING", {
                    "default": """void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5*cos(iTime + uv.xyx*6.283 + vec3(0,2,4));
    fragColor = vec4(col, 1.0);
}""",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def add(self, pipeline, shader_type, code):
        from .shaderflow_bridge import HeadlessGLSLRenderer
        pipe = _clone_pipeline(pipeline)

        if shader_type == "shadertoy":
            full_code = HeadlessGLSLRenderer.wrap_shadertoy(code)
        else:
            full_code = code

        layer = {
            "type": "glsl",
            "params": {"fragment_shader": full_code},
            "_glsl_renderer": None,
        }
        pipe["layers"].append(layer)
        logger.info(f"[SF Pipeline] Added GLSL layer ({shader_type})")
        return (pipe,)


# ---------------------------------------------------------------------------
# SF_AddDepthFlow — add DepthFlow parallax layer to pipeline
# ---------------------------------------------------------------------------

class SF_AddDepthFlow:
    """Add a DepthFlow 2.5D parallax effect layer to the pipeline.
    Uses pipeline background image as the source.
    Supports audio-reactive parallax when spectrum is available.

    CUDA renderer is preferred; falls back to subprocess if unavailable.
    Existing DepthFlowGenerator node is NOT modified.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_SF_PIPELINE,),
                "camera_movement": (["vertical", "horizontal", "zoom",
                                     "circle", "dolly", "orbital", "static"], {
                    "default": "vertical",
                }),
                "movement_intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
                "steady_depth": ("FLOAT", {
                    "default": 0.3, "min": -1.0, "max": 2.0, "step": 0.05,
                }),
                "isometric": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            },
            "optional": {
                "depth_map": ("IMAGE", {
                    "tooltip": "外部深度图（不提供则自动估算）",
                }),
                "depth_estimator": (["da2", "da1", "depthpro", "zoedepth"], {
                    "default": "da2",
                }),
                "movement_smooth": ("BOOLEAN", {"default": True}),
                "movement_loop": ("BOOLEAN", {"default": True}),
                "movement_reverse": ("BOOLEAN", {"default": False}),
                "movement_phase": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "ssaa": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.25,
                }),
                "audio_preset": (["none", "subtle_pulse", "heartbeat_zoom", "aggressive_bounce", "chaotic_shake", "custom"], {
                    "default": "none",
                    "tooltip": "音频驱动预设模式",
                }),
                "audio_target": (["zoom", "height", "both", "isometric", "phase"], {
                    "default": "both",
                    "tooltip": "自定义模式下的音频驱动目标",
                }),
                "audio_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "音频驱动强度缩放",
                }),
            },
        }

    RETURN_TYPES = (T_SF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow Pipeline"

    def add(self, pipeline, camera_movement, movement_intensity,
            steady_depth, isometric,
            depth_map=None, depth_estimator="da2",
            movement_smooth=True, movement_loop=True,
            movement_reverse=False, movement_phase=0.0,
            ssaa=1.0, audio_preset="none", audio_target="both", audio_scale=1.5):
        pipe = _clone_pipeline(pipeline)

        if pipe["background"] is None:
            raise ValueError("SF_AddDepthFlow requires a background image in the pipeline. "
                             "Connect an IMAGE to SF_Pipeline.")

        # Extract source image (first frame of background)
        bg = pipe["background"]  # (N, H, W, 3) uint8
        src_img = bg[0]  # (H, W, 3) uint8

        # Process depth map
        depth_np = None
        if depth_map is not None:
            if HAS_TORCH and isinstance(depth_map, torch.Tensor):
                d = depth_map.cpu().numpy()
            else:
                d = np.asarray(depth_map)
            if d.ndim == 4:
                d = d[0]
            if d.ndim == 3:
                d = np.mean(d, axis=-1) if d.shape[-1] in (3, 4) else d[:, :, 0]
            depth_np = d.astype(np.float32)
            if depth_np.max() > 1.5:
                depth_np = depth_np / 255.0
            logger.info(f"[SF Pipeline] DepthFlow: using provided depth map {depth_np.shape}")

        layer = {
            "type": "depthflow",
            "params": {
                "camera_movement": camera_movement,
                "movement_intensity": movement_intensity,
                "steady_depth": steady_depth,
                "isometric": isometric,
                "movement_smooth": movement_smooth,
                "movement_loop": movement_loop,
                "movement_reverse": movement_reverse,
                "movement_phase": movement_phase,
                "ssaa": ssaa,
                "depth_estimator": depth_estimator,
                "audio_preset": audio_preset,
                "audio_target": audio_target,
                "audio_scale": audio_scale,
            },
            "_src_img": src_img,
            "_depth_np": depth_np,
            "_renderer": None,  # initialized at render time
            "spectrum": pipe.get("spectrum"),  # for audio-reactive
        }

        pipe["layers"].append(layer)
        logger.info(f"[SF Pipeline] Added DepthFlow layer: {camera_movement}, "
                    f"intensity={movement_intensity}")
        return (pipe,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_SF_Pipeline": SF_Pipeline,
    "VHS_SF_AddSpectrum": SF_AddSpectrum,
    "VHS_SF_AddVisualizer": SF_AddVisualizer,
    "VHS_SF_AddEffect": SF_AddEffect,
    "VHS_SF_AddGLSL": SF_AddGLSL,
    "VHS_SF_AddDepthFlow": SF_AddDepthFlow,
    "VHS_SF_PipelineRender": SF_PipelineRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SF_Pipeline": "SF Pipeline 🔗🅥🅗🅢",
    "VHS_SF_AddSpectrum": "SF Add Spectrum 📊🅥🅗🅢",
    "VHS_SF_AddVisualizer": "SF Add Visualizer 🎨🅥🅗🅢",
    "VHS_SF_AddEffect": "SF Add Effect ✨🅥🅗🅢",
    "VHS_SF_AddGLSL": "SF Add GLSL 🔧🅥🅗🅢",
    "VHS_SF_AddDepthFlow": "SF Add DepthFlow 🌊🅥🅗🅢",
    "VHS_SF_PipelineRender": "SF Pipeline Render 🎬🅥🅗🅢",
}

