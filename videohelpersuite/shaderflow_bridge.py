"""
ShaderFlow → ComfyUI Bridge Layer
Provides headless OpenGL (EGL) rendering with PyTorch fallback.
Route B (EGL/ModernGL) is preferred; Route A (PyTorch) is the fallback.
"""
import os
import sys
import math
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger("ShaderFlowBridge")

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------

_EGL_AVAILABLE: Optional[bool] = None
_SHADERFLOW_AVAILABLE: Optional[bool] = None
_TORCH_AVAILABLE: Optional[bool] = None


def _check_egl() -> bool:
    """Check if EGL headless OpenGL context can be created."""
    global _EGL_AVAILABLE
    if _EGL_AVAILABLE is not None:
        return _EGL_AVAILABLE
    _EGL_AVAILABLE = False
    # macOS never supports EGL
    if sys.platform == "darwin":
        return False
    try:
        # Try to create a headless moderngl context via EGL
        import moderngl
        ctx = moderngl.create_standalone_context(backend="egl")
        info = ctx.info
        ctx.release()
        _EGL_AVAILABLE = True
        logger.info(f"EGL headless OpenGL available: {info.get('GL_RENDERER', 'unknown')}")
    except Exception as e:
        logger.info(f"EGL headless OpenGL not available: {e}")
        # Try OSMesa as alternative
        try:
            import moderngl
            ctx = moderngl.create_standalone_context()
            ctx.release()
            _EGL_AVAILABLE = True
            logger.info("Standalone OpenGL context available (non-EGL)")
        except Exception as e2:
            logger.info(f"No headless OpenGL available: {e2}")
    return _EGL_AVAILABLE


def _check_shaderflow() -> bool:
    """Check if ShaderFlow package is importable."""
    global _SHADERFLOW_AVAILABLE
    if _SHADERFLOW_AVAILABLE is not None:
        return _SHADERFLOW_AVAILABLE
    _SHADERFLOW_AVAILABLE = False
    try:
        import shaderflow
        _SHADERFLOW_AVAILABLE = True
    except ImportError:
        # Try adding the local ShaderFlow path
        sf_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "ShaderFlow",
            Path("/Volumes/GodLin/Dev/Codes/ComfyUI/own/ShaderFlow"),
        ]
        for p in sf_paths:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            import shaderflow
            _SHADERFLOW_AVAILABLE = True
            logger.info(f"ShaderFlow found at {shaderflow.__file__}")
        except ImportError:
            pass
    return _SHADERFLOW_AVAILABLE


def _check_torch() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        import torch
        _TORCH_AVAILABLE = True
    except ImportError:
        _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def get_available_backend() -> str:
    """Return best available backend: 'shaderflow', 'torch', or 'none'."""
    if _check_egl() and _check_shaderflow():
        return "shaderflow"
    if _check_torch():
        return "torch"
    return "none"


# ---------------------------------------------------------------------------
# DynamicNumber: second-order system (ported from ShaderFlow dynamics.py)
# ---------------------------------------------------------------------------

class DynamicNumber:
    """Second-order system for smooth animation (spring-damper model).
    Direct port of ShaderFlow's DynamicNumber without ModernGL dependency.
    """

    def __init__(
        self,
        value: float = 0.0,
        frequency: float = 1.0,
        zeta: float = 1.0,
        response: float = 0.0,
        dtype=np.float32,
    ):
        self.frequency = frequency
        self.zeta = zeta
        self.response = response
        self.dtype = dtype

        self._value = np.atleast_1d(np.array(value, dtype=dtype))
        self._target = self._value.copy()
        self._velocity = np.zeros_like(self._value)
        self._recalc_constants()

    def _recalc_constants(self):
        w = 2.0 * math.pi * self.frequency
        self._k1 = self.zeta / (math.pi * self.frequency) if self.frequency > 0 else 0
        self._k2 = 1.0 / (w * w) if w > 0 else 0
        self._k3 = (self.response * self.zeta) / w if w > 0 else 0

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, v):
        self._value = np.atleast_1d(np.array(v, dtype=self.dtype))

    @property
    def target(self) -> np.ndarray:
        return self._target

    @target.setter
    def target(self, v):
        self._target = np.atleast_1d(np.array(v, dtype=self.dtype))

    def set(self, v):
        self._value = np.atleast_1d(np.array(v, dtype=self.dtype))
        self._target = self._value.copy()
        self._velocity = np.zeros_like(self._value)

    def next(self, dt: float):
        if dt <= 0:
            return self._value
        # Semi-implicit Euler integration
        self._recalc_constants()
        k2_stable = max(self._k2, 1.1 * (dt * dt / 4 + dt * self._k1 / 2))
        accel = (self._target - self._value - self._k1 * self._velocity) / k2_stable
        self._velocity = self._velocity + dt * accel
        self._value = self._value + dt * self._velocity
        return self._value


# ---------------------------------------------------------------------------
# FFT Spectrogram Engine (no OpenGL dependency)
# ---------------------------------------------------------------------------

class SpectrogramEngine:
    """Numpy-only FFT spectrogram, ported from ShaderFlow BrokenSpectrogram."""

    def __init__(
        self,
        samplerate: int = 44100,
        fft_n: int = 12,
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        bins: int = 512,
        channels: int = 2,
    ):
        self.samplerate = samplerate
        self.fft_n = fft_n
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bins = bins
        self.channels = channels
        self._matrix = None

    @property
    def fft_size(self) -> int:
        return int(2 ** self.fft_n)

    @property
    def fft_bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def fft_frequencies(self) -> np.ndarray:
        return np.fft.rfftfreq(self.fft_size, 1.0 / self.samplerate)

    def _octave_scale(self, x):
        return np.log2(x)

    def _octave_inv(self, x):
        return 2.0 ** x

    @property
    def center_frequencies(self) -> np.ndarray:
        return self._octave_inv(np.linspace(
            self._octave_scale(self.min_freq),
            self._octave_scale(self.max_freq),
            self.bins,
        ))

    def _build_matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        freq_step = self.fft_frequencies[1] if len(self.fft_frequencies) > 1 else 1.0
        indices = self.center_frequencies / freq_step
        # Euler interpolation kernel (from ShaderFlow)
        end = 1.2
        matrix = np.zeros((self.bins, self.fft_bins), dtype=np.float32)
        for i, idx in enumerate(indices):
            x = idx - np.arange(self.fft_bins)
            kernel = np.exp(-((2 * x / end) ** 2)) / (end * (math.pi ** 0.5))
            kernel[np.abs(kernel) < 1e-5] = 0
            matrix[i] = kernel
        self._matrix = matrix
        return matrix

    def compute(self, samples: np.ndarray) -> np.ndarray:
        """Compute spectrogram from raw audio samples.
        Args:
            samples: shape (channels, fft_size) or (fft_size,)
        Returns:
            shape (channels, bins) or (bins,)
        """
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        # Pad or trim to fft_size
        if samples.shape[-1] < self.fft_size:
            pad = self.fft_size - samples.shape[-1]
            samples = np.pad(samples, ((0, 0), (0, pad)))
        elif samples.shape[-1] > self.fft_size:
            samples = samples[:, -self.fft_size:]

        window = np.hanning(self.fft_size)
        windowed = window * samples
        fft_data = np.abs(np.fft.rfft(windowed)) ** 2  # Power spectrum
        matrix = self._build_matrix()
        result = np.sqrt(matrix @ fft_data.T).T  # (channels, bins)
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio file reader (ffmpeg-based, no soundcard dependency)
# ---------------------------------------------------------------------------

def read_audio_file(
    path: str,
    samplerate: int = 44100,
    channels: int = 2,
    dtype=np.float32,
) -> Tuple[np.ndarray, int]:
    """Read audio file via ffmpeg, returns (samples, samplerate).
    samples shape: (channels, total_samples)
    """
    import subprocess
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", str(channels),
        "-ar", str(samplerate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {proc.stderr.decode()[:500]}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    # Reshape to (total_frames, channels) then transpose
    total_frames = len(raw) // channels
    raw = raw[: total_frames * channels].reshape(total_frames, channels).T
    return raw, samplerate


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# PyTorch Visualizer Renderer (Route A fallback)
# ---------------------------------------------------------------------------

class TorchVisualizerRenderer:
    """Pure PyTorch/numpy renderer for audio visualizations.
    Used when EGL/ShaderFlow is not available.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    def render_bars_frame(
        self,
        spectrum: np.ndarray,
        bg_color: Tuple[int, int, int] = (10, 10, 25),
        bar_colors: Optional[List[Tuple[int, int, int]]] = None,
        glow: bool = True,
        mirror: bool = True,
    ) -> np.ndarray:
        """Render a single bars visualization frame.
        Args:
            spectrum: shape (bins,) normalized 0-1
            bg_color: background RGB
            bar_colors: gradient colors for bars
        Returns:
            shape (height, width, 3) uint8
        """
        frame = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        bins = len(spectrum)
        if bins == 0:
            return frame

        # Default neon gradient: cyan → magenta → orange
        if bar_colors is None:
            bar_colors = [
                (0, 255, 220),   # cyan
                (120, 80, 255),  # purple
                (255, 50, 180),  # magenta
                (255, 160, 40),  # orange
            ]

        bar_area_w = int(self.width * 0.9)
        bar_area_x = (self.width - bar_area_w) // 2
        bar_w = max(1, bar_area_w // bins)
        gap = max(0, (bar_area_w - bar_w * bins) // max(1, bins - 1))
        actual_bar_w = max(1, bar_w - 1)

        max_bar_h = int(self.height * 0.40)
        center_y = self.height // 2

        for i in range(bins):
            val = float(np.clip(spectrum[i], 0, 1))
            h = int(val * max_bar_h)
            if h < 1:
                continue
            x = bar_area_x + i * (bar_w + gap)
            if x + actual_bar_w > self.width:
                break

            # Interpolate color
            t = i / max(1, bins - 1)
            color = self._gradient_color(t, bar_colors)

            # Draw bar upward from center
            y_top = center_y - h
            frame[y_top:center_y, x:x + actual_bar_w] = color

            # Mirror downward
            if mirror:
                y_bot = center_y + h
                frame[center_y:min(y_bot, self.height), x:x + actual_bar_w] = color

        # Simple glow effect via blur blend
        if glow:
            try:
                import cv2
                blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=15, sigmaY=15)
                frame = np.clip(frame.astype(np.float32) + blurred.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
            except ImportError:
                pass
        return frame

    def render_waveform_frame(
        self,
        waveform: np.ndarray,
        bg_color: Tuple[int, int, int] = (10, 10, 25),
        line_color: Tuple[int, int, int] = (0, 255, 180),
    ) -> np.ndarray:
        """Render waveform oscilloscope frame."""
        frame = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        if len(waveform) == 0:
            return frame
        try:
            import cv2
        except ImportError:
            return frame

        center_y = self.height // 2
        amp = self.height * 0.35
        n_points = min(len(waveform), self.width)
        indices = np.linspace(0, len(waveform) - 1, n_points).astype(int)
        sampled = waveform[indices]

        points = np.zeros((n_points, 1, 2), dtype=np.int32)
        for i in range(n_points):
            x = int(i * self.width / n_points)
            y = int(center_y - sampled[i] * amp)
            y = np.clip(y, 0, self.height - 1)
            points[i, 0] = [x, y]

        cv2.polylines(frame, [points], False, line_color, 2, cv2.LINE_AA)
        # Glow
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=8)
        frame = np.clip(frame.astype(np.float32) + blurred.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        return frame

    def render_radial_frame(
        self,
        spectrum: np.ndarray,
        bg_color: Tuple[int, int, int] = (10, 10, 25),
        bar_colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Render radial/circular bars visualization."""
        frame = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        try:
            import cv2
        except ImportError:
            return frame

        if bar_colors is None:
            bar_colors = [(0, 255, 220), (120, 80, 255), (255, 50, 180), (255, 160, 40)]

        bins = len(spectrum)
        if bins == 0:
            return frame

        cx, cy = self.width // 2, self.height // 2
        r_inner = min(cx, cy) * 0.2
        r_max = min(cx, cy) * 0.7

        for i in range(bins):
            angle = 2.0 * math.pi * i / bins - math.pi / 2
            val = float(np.clip(spectrum[i], 0, 1))
            r_outer = r_inner + val * (r_max - r_inner)
            x1 = int(cx + r_inner * math.cos(angle))
            y1 = int(cy + r_inner * math.sin(angle))
            x2 = int(cx + r_outer * math.cos(angle))
            y2 = int(cy + r_outer * math.sin(angle))
            color = self._gradient_color(i / max(1, bins - 1), bar_colors)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=12)
        frame = np.clip(frame.astype(np.float32) + blurred.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _gradient_color(
        t: float,
        colors: List[Tuple[int, int, int]],
    ) -> Tuple[int, int, int]:
        """Interpolate through a list of colors at position t in [0,1]."""
        t = max(0.0, min(1.0, t))
        n = len(colors)
        if n == 0:
            return (255, 255, 255)
        if n == 1:
            return colors[0]
        idx = t * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return tuple(int(colors[lo][c] * (1 - frac) + colors[hi][c] * frac) for c in range(3))

    def composite_with_background(
        self,
        vis_frame: np.ndarray,
        bg_image: Optional[np.ndarray] = None,
        bg_opacity: float = 0.3,
    ) -> np.ndarray:
        """Composite visualization frame over a background image."""
        if bg_image is None:
            return vis_frame
        try:
            import cv2
            bg = cv2.resize(bg_image, (self.width, self.height))
            if bg.shape[2] == 4:
                bg = bg[:, :, :3]
            result = np.clip(
                bg.astype(np.float32) * bg_opacity
                + vis_frame.astype(np.float32),
                0, 255
            ).astype(np.uint8)
            return result
        except ImportError:
            return vis_frame


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def write_frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    audio_path: Optional[str] = None,
    codec: str = "libx264",
    quality: int = 23,
) -> str:
    """Write frames to video file via ffmpeg pipe, optionally mux audio."""
    h, w = frames[0].shape[:2]
    tmp_video = output_path + ".tmp.mp4" if audio_path else output_path

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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
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
        os.remove(tmp_video)
    return output_path
