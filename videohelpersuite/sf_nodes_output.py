"""
ShaderFlow Modular Nodes — Output / Render nodes.
SF_RenderToVideo: the ONLY node that actually renders frames.
Streams frames to ffmpeg one-by-one — memory O(1), not O(N).
"""
import os
import hashlib
import logging
import tempfile
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from .sf_core import (
    T_SF_AUDIO, T_SF_CANVAS, T_SF_LAYER,
    render_layer_frame, get_total_frames,
    logger,
)
from .shaderflow_bridge import (
    HeadlessGLSLRenderer,
    DynamicNumber,
    write_frames_to_video_with_progress,
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


def _output_dir() -> str:
    if folder_paths:
        return folder_paths.get_output_directory()
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Pre-render initialization (GLSL context, key dynamics, etc.)
# ---------------------------------------------------------------------------

def _prepare_layer_for_render(layer: Dict, canvas: Dict):
    """Initialize runtime resources before rendering starts."""
    layer_type = layer.get("type", "")

    # GLSL: create and compile renderer
    if layer_type == "glsl":
        shader_code = layer.get("params", {}).get("fragment_shader", "")
        if shader_code and layer.get("_glsl_renderer") is None:
            from .shaderflow_bridge import HeadlessGLSLRenderer, get_available_backend
            backend = get_available_backend()
            if backend in ("shaderflow", "torch"):
                renderer = HeadlessGLSLRenderer(
                    width=canvas["width"], height=canvas["height"]
                )
                if renderer.compile(shader_code):
                    layer["_glsl_renderer"] = renderer
                    logger.info("[SF] GLSL renderer initialized (headless OpenGL)")
                else:
                    logger.warning("[SF] GLSL compile failed, using procedural fallback")
            else:
                logger.info("[SF] No OpenGL backend, using procedural fallback")

    # Piano roll: pre-compute key dynamics
    elif layer_type == "piano_roll":
        midi = layer.get("midi")
        params = layer.get("params", {})
        if midi and params.get("smooth_keys", True):
            fps = canvas["fps"]
            total = get_total_frames(canvas)
            notes = midi["notes"]
            max_note = midi.get("max_note", 108) + 1

            logger.info(f"[SF] Pre-computing piano key dynamics: {total} frames, {max_note} keys")
            dynamics = [DynamicNumber(0.0, frequency=8.0, zeta=0.5) for _ in range(max_note)]
            cache = []

            for fi in range(total):
                t = fi / fps
                dt = 1.0 / fps
                targets = np.zeros(max_note, dtype=np.float32)
                for n in notes:
                    nn = n["note"]
                    if nn < max_note and n["start"] <= t <= n["end"]:
                        targets[nn] = float(n.get("velocity", 100))

                kd = np.zeros(max_note, dtype=np.float32)
                for k in range(max_note):
                    dynamics[k].target = targets[k]
                    kd[k] = float(dynamics[k].next(dt))
                cache.append(kd)

            layer["_key_dynamics_cache"] = cache

    # Waveform: inject fps from canvas
    elif layer_type == "waveform":
        layer["fps"] = canvas["fps"]

    # Recursive: prepare source layers for chained FX
    if "source_layer" in layer:
        _prepare_layer_for_render(layer["source_layer"], canvas)


def _cleanup_layer(layer: Dict):
    """Release runtime resources after rendering."""
    if layer.get("type") == "glsl" and layer.get("_glsl_renderer"):
        layer["_glsl_renderer"].release()
        layer["_glsl_renderer"] = None
    if "source_layer" in layer:
        _cleanup_layer(layer["source_layer"])


# ---------------------------------------------------------------------------
# SF_RenderToVideo — the main output node
# ---------------------------------------------------------------------------

class SF_RenderToVideo:
    """Render SF_LAYER to video file via streaming ffmpeg.
    Memory usage: O(1) per frame — frames are piped directly to ffmpeg.
    Optional: output IMAGE tensor for preview (warning: high memory).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_layer": (T_SF_LAYER,),
                "sf_canvas": (T_SF_CANVAS,),
                "output_prefix": ("STRING", {
                    "default": "sf_render_",
                    "tooltip": "输出文件名前缀",
                }),
                "output_format": (["mp4", "mkv", "webm"], {
                    "default": "mp4",
                }),
                "codec": (["libx264", "libx265", "libvpx-vp9"], {
                    "default": "libx264",
                }),
                "quality": ("INT", {
                    "default": 23, "min": 0, "max": 51,
                    "tooltip": "CRF 质量 (0=无损, 23=默认, 51=最低)",
                }),
                "output_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否同时输出 IMAGE tensor（⚠️ 高内存：1920x1080x3600帧≈22GB）",
                }),
            },
            "optional": {
                "sf_audio": (T_SF_AUDIO, {
                    "tooltip": "音频输入，将 mux 到输出视频中",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "外部音频文件路径（sf_audio 优先）",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Output"

    def render(self, sf_layer, sf_canvas, output_prefix, output_format,
               codec, quality, output_frames, sf_audio=None, audio_path=""):

        w = sf_canvas["width"]
        h = sf_canvas["height"]
        fps = sf_canvas["fps"]
        total = get_total_frames(sf_canvas)

        logger.info(f"[SF] ═══ Render Start ═══")
        logger.info(f"[SF] Canvas: {w}x{h} @ {fps}fps, {total} frames ({sf_canvas['duration']:.2f}s)")
        logger.info(f"[SF] Layer type: {sf_layer.get('type', '?')}")
        logger.info(f"[SF] Output: {output_format}, codec={codec}, crf={quality}")
        logger.info(f"[SF] Output frames tensor: {output_frames}")

        # Resolve audio path
        resolved_audio = None
        if sf_audio is not None:
            # Write SF_AUDIO to temp wav
            resolved_audio = self._audio_to_wav(sf_audio)
        elif audio_path and audio_path.strip() and os.path.exists(audio_path.strip()):
            resolved_audio = audio_path.strip()

        # Prepare layer runtime resources
        _prepare_layer_for_render(sf_layer, sf_canvas)

        # Setup output path
        out_dir = _output_dir()
        layer_hash = hashlib.md5(str(sf_layer.get("type", "")).encode()).hexdigest()[:6]
        out_path = os.path.join(out_dir, f"{output_prefix}{layer_hash}.{output_format}")

        # Setup ffmpeg pipe
        tmp_video = out_path + ".tmp.mp4" if resolved_audio else out_path
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec, "-crf", str(quality),
            "-pix_fmt", "yuv420p",
            tmp_video,
        ]

        # Progress: render=0-85%, encode-flush=85-95%, mux=95-100%
        pbar = ProgressBar(100) if ProgressBar else None
        collected_frames = [] if output_frames else None

        logger.info(f"[SF] Starting ffmpeg pipe: {' '.join(cmd[:5])}...")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        try:
            for fi in range(total):
                frame = render_layer_frame(sf_layer, fi, sf_canvas)

                # Ensure correct shape
                if frame.shape[0] != h or frame.shape[1] != w:
                    try:
                        import cv2
                        frame = cv2.resize(frame, (w, h))
                    except ImportError:
                        frame = np.zeros((h, w, 3), dtype=np.uint8)

                proc.stdin.write(frame.tobytes())

                if collected_frames is not None:
                    collected_frames.append(frame)

                # Progress update
                if pbar:
                    progress = int(fi / total * 85)
                    pbar.update_absolute(progress)

                # Log every 10%
                if fi > 0 and fi % max(1, total // 10) == 0:
                    pct = fi / total * 100
                    logger.info(f"[SF] Rendering: frame {fi}/{total} ({pct:.0f}%)")

        finally:
            proc.stdin.close()

        if pbar:
            pbar.update_absolute(85)
        logger.info(f"[SF] All frames piped, waiting for ffmpeg flush...")

        proc.wait()
        if pbar:
            pbar.update_absolute(95)

        if proc.returncode != 0:
            _cleanup_layer(sf_layer)
            raise RuntimeError(f"ffmpeg encoding failed (exit {proc.returncode})")

        # Mux audio if needed
        if resolved_audio:
            logger.info(f"[SF] Muxing audio: {resolved_audio}")
            mux_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", tmp_video,
                "-i", resolved_audio,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path,
            ]
            subprocess.run(mux_cmd, check=True, timeout=300)
            try:
                os.remove(tmp_video)
            except OSError:
                pass

        if pbar:
            pbar.update_absolute(100)

        # Cleanup runtime resources
        _cleanup_layer(sf_layer)

        file_size = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"[SF] ═══ Render Complete ═══")
        logger.info(f"[SF] Output: {out_path} ({file_size:.1f} MB)")

        # Build IMAGE tensor if requested
        frames_tensor = None
        if collected_frames:
            arr = np.stack(collected_frames, axis=0).astype(np.float32) / 255.0
            if HAS_TORCH:
                frames_tensor = torch.from_numpy(arr)
            else:
                frames_tensor = arr
            logger.info(f"[SF] Frames tensor: shape={arr.shape}, {arr.nbytes / 1024 / 1024:.1f} MB")

        return (out_path, frames_tensor)

    def _audio_to_wav(self, sf_audio: Dict) -> str:
        """Write SF_AUDIO samples to a temp WAV file for muxing."""
        import wave
        samples = sf_audio["samples"]
        sr = sf_audio["sr"]

        # Convert to int16
        if samples.dtype == np.float32 or samples.dtype == np.float64:
            int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        else:
            int_samples = samples.astype(np.int16)

        # Interleave channels: (C, N) → (N, C)
        if int_samples.ndim == 2:
            int_samples = int_samples.T

        tmp_dir = _output_dir()
        wav_path = os.path.join(tmp_dir, f"sf_audio_tmp_{id(sf_audio) % 99999}.wav")
        with wave.open(wav_path, "w") as wf:
            n_channels = sf_audio.get("channels", 2)
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_samples.tobytes())

        logger.info(f"[SF] Audio written to WAV: {wav_path}")
        return wav_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_SF_RenderToVideo": SF_RenderToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SF_RenderToVideo": "SF Render to Video 🎬🅥🅗🅢",
}
