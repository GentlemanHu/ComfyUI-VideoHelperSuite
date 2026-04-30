"""
ShaderFlow Modular Nodes — Analysis & Processing nodes.
SF_AudioToSpectrum, SF_SmoothCurve, SF_WaveGenerator,
SF_CurveToFloat, SF_SpectrumToAmplitude
"""
import math
import logging
from typing import Any, Dict

import numpy as np

from .sf_core import (
    T_SF_AUDIO, T_SF_SPECTRUM, T_SF_CURVE, T_SF_CANVAS,
    logger,
)
from .shaderflow_bridge import (
    compute_spectrum_batch,
    DynamicNumber,
)

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


# ---------------------------------------------------------------------------
# SF_AudioToSpectrum
# ---------------------------------------------------------------------------

class SF_AudioToSpectrum:
    """Compute per-frame FFT spectrum from audio → SF_SPECTRUM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_audio": (T_SF_AUDIO,),
                "sf_canvas": (T_SF_CANVAS, {
                    "tooltip": "用于获取 fps 和计算总帧数",
                }),
                "fft_power": ("INT", {
                    "default": 12, "min": 8, "max": 16,
                    "tooltip": "FFT 窗口大小 = 2^N。12=4096, 13=8192",
                }),
                "spectrum_bins": ("INT", {
                    "default": 128, "min": 16, "max": 1024, "step": 16,
                    "tooltip": "频谱柱数",
                }),
                "min_freq": ("FLOAT", {
                    "default": 20.0, "min": 10.0, "max": 2000.0,
                    "tooltip": "最低频率 Hz",
                }),
                "max_freq": ("FLOAT", {
                    "default": 16000.0, "min": 1000.0, "max": 22000.0,
                    "tooltip": "最高频率 Hz",
                }),
            },
        }

    RETURN_TYPES = (T_SF_SPECTRUM,)
    RETURN_NAMES = ("sf_spectrum",)
    FUNCTION = "compute"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Analysis"

    def compute(self, sf_audio, sf_canvas, fft_power, spectrum_bins,
                min_freq, max_freq):
        fps = sf_canvas["fps"]
        pbar = ProgressBar(100) if ProgressBar else None

        def progress_cb(current, total):
            if pbar and total > 0:
                pbar.update_absolute(int(current / total * 100))

        spectrum = compute_spectrum_batch(
            audio_data=sf_audio,
            fps=fps,
            fft_n=fft_power,
            bins=spectrum_bins,
            min_freq=min_freq,
            max_freq=max_freq,
            progress_cb=progress_cb,
        )
        return (spectrum,)


# ---------------------------------------------------------------------------
# SF_SmoothCurve — second-order dynamics on spectrum or curve
# ---------------------------------------------------------------------------

class SF_SmoothCurve:
    """Apply second-order dynamics smoothing to SF_SPECTRUM or SF_CURVE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency": ("FLOAT", {
                    "default": 4.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "动力学频率 — 值越大跟踪越快",
                }),
                "zeta": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 3.0, "step": 0.05,
                    "tooltip": "阻尼系数 — <1弹性 | 1临界阻尼 | >1过阻尼",
                }),
                "response": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "初始响应加速度",
                }),
            },
            "optional": {
                "sf_spectrum": (T_SF_SPECTRUM, {
                    "tooltip": "频谱数据输入",
                }),
                "sf_curve": (T_SF_CURVE, {
                    "tooltip": "曲线数据输入",
                }),
            },
        }

    RETURN_TYPES = (T_SF_SPECTRUM, T_SF_CURVE)
    RETURN_NAMES = ("sf_spectrum", "sf_curve")
    FUNCTION = "smooth"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Analysis"

    def smooth(self, frequency, zeta, response,
               sf_spectrum=None, sf_curve=None):
        out_spectrum = None
        out_curve = None

        if sf_spectrum is not None:
            out_spectrum = self._smooth_spectrum(sf_spectrum, frequency, zeta, response)

        if sf_curve is not None:
            out_curve = self._smooth_curve(sf_curve, frequency, zeta, response)

        return (out_spectrum, out_curve)

    def _smooth_spectrum(self, spectrum, freq, zeta, response):
        bins_data = spectrum["bins"]  # (F, B)
        fps = spectrum["fps"]
        dt = 1.0 / fps
        n_frames, n_bins = bins_data.shape

        logger.info(f"[SF] Smoothing spectrum: {n_frames}x{n_bins}, freq={freq}, zeta={zeta}")

        dynamics = [DynamicNumber(0.0, frequency=freq, zeta=zeta, response=response)
                    for _ in range(n_bins)]
        smoothed = np.zeros_like(bins_data)

        for fi in range(n_frames):
            for bi in range(n_bins):
                dynamics[bi].target = bins_data[fi, bi]
                smoothed[fi, bi] = max(0.0, dynamics[bi].next_scalar(dt))

        result = dict(spectrum)
        result["bins"] = smoothed
        return result

    def _smooth_curve(self, curve, freq, zeta, response):
        values = curve["values"]
        fps = curve["fps"]
        dt = 1.0 / fps

        logger.info(f"[SF] Smoothing curve: {len(values)} values, freq={freq}, zeta={zeta}")

        dyn = DynamicNumber(values[0] if values else 0.0,
                            frequency=freq, zeta=zeta, response=response)
        smoothed = []
        for v in values:
            dyn.target = v
            smoothed.append(dyn.next_scalar(dt))

        result = dict(curve)
        result["values"] = smoothed
        return result


# ---------------------------------------------------------------------------
# SF_WaveGenerator
# ---------------------------------------------------------------------------

class SF_WaveGenerator:
    """Generate periodic wave → SF_CURVE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_canvas": (T_SF_CANVAS,),
                "wave_type": (["sine", "square", "triangle", "sawtooth", "pulse"], {
                    "default": "sine",
                }),
                "frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1,
                    "tooltip": "波形频率 Hz",
                }),
                "amplitude": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
                "offset": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                }),
                "phase": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 6.283, "step": 0.1,
                    "tooltip": "初始相位 (弧度)",
                }),
            },
        }

    RETURN_TYPES = (T_SF_CURVE,)
    RETURN_NAMES = ("sf_curve",)
    FUNCTION = "generate"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Analysis"

    def generate(self, sf_canvas, wave_type, frequency, amplitude, offset, phase):
        fps = sf_canvas["fps"]
        duration = sf_canvas["duration"]
        total = max(1, int(duration * fps))

        values = []
        for fi in range(total):
            t = fi / fps
            angle = 2.0 * math.pi * frequency * t + phase

            if wave_type == "sine":
                v = math.sin(angle)
            elif wave_type == "square":
                v = 1.0 if math.sin(angle) >= 0 else -1.0
            elif wave_type == "triangle":
                v = 2.0 * abs(2.0 * (angle / (2.0 * math.pi) % 1.0) - 1.0) - 1.0
            elif wave_type == "sawtooth":
                v = 2.0 * (angle / (2.0 * math.pi) % 1.0) - 1.0
            elif wave_type == "pulse":
                v = 1.0 if (angle / (2.0 * math.pi) % 1.0) < 0.1 else 0.0
            else:
                v = 0.0

            values.append(v * amplitude + offset)

        logger.info(f"[SF] Wave generated: {wave_type}, {frequency}Hz, {total} frames")
        return ({"values": values, "fps": fps},)


# ---------------------------------------------------------------------------
# SF_CurveToFloat
# ---------------------------------------------------------------------------

class SF_CurveToFloat:
    """Extract single float value from SF_CURVE at a given frame index.
    Useful for driving DepthFlow or other scalar parameter nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_curve": (T_SF_CURVE,),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "帧索引",
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "extract"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Analysis"

    def extract(self, sf_curve, frame_index):
        values = sf_curve["values"]
        idx = min(frame_index, len(values) - 1) if values else 0
        val = values[idx] if values else 0.0
        return (float(val),)


# ---------------------------------------------------------------------------
# SF_SpectrumToAmplitude
# ---------------------------------------------------------------------------

class SF_SpectrumToAmplitude:
    """Extract amplitude curve from specific frequency band of SF_SPECTRUM → SF_CURVE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_spectrum": (T_SF_SPECTRUM,),
                "band": (["full", "low", "mid", "high", "sub_bass", "presence"], {
                    "default": "full",
                    "tooltip": "频段选择: full=全频, low=低频(0-25%), mid=中频(25-60%), high=高频(60-100%)",
                }),
                "aggregation": (["mean", "max", "rms"], {
                    "default": "mean",
                    "tooltip": "聚合方式",
                }),
            },
        }

    RETURN_TYPES = (T_SF_CURVE,)
    RETURN_NAMES = ("sf_curve",)
    FUNCTION = "extract"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/ShaderFlow/Analysis"

    def extract(self, sf_spectrum, band, aggregation):
        bins_data = sf_spectrum["bins"]  # (F, B)
        fps = sf_spectrum["fps"]
        n_frames, n_bins = bins_data.shape

        # Define band ranges (as fraction of total bins)
        band_ranges = {
            "full": (0.0, 1.0),
            "sub_bass": (0.0, 0.08),
            "low": (0.0, 0.25),
            "mid": (0.25, 0.60),
            "high": (0.60, 1.0),
            "presence": (0.40, 0.75),
        }
        lo_frac, hi_frac = band_ranges.get(band, (0.0, 1.0))
        lo_bin = int(lo_frac * n_bins)
        hi_bin = max(lo_bin + 1, int(hi_frac * n_bins))

        values = []
        for fi in range(n_frames):
            segment = bins_data[fi, lo_bin:hi_bin]
            if aggregation == "mean":
                v = float(np.mean(segment))
            elif aggregation == "max":
                v = float(np.max(segment))
            elif aggregation == "rms":
                v = float(np.sqrt(np.mean(segment ** 2)))
            else:
                v = float(np.mean(segment))
            values.append(v)

        logger.info(
            f"[SF] Spectrum→Amplitude: band={band}[{lo_bin}:{hi_bin}], "
            f"agg={aggregation}, {n_frames} frames"
        )
        return ({"values": values, "fps": fps},)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_SF_AudioToSpectrum": SF_AudioToSpectrum,
    "VHS_SF_SmoothCurve": SF_SmoothCurve,
    "VHS_SF_WaveGenerator": SF_WaveGenerator,
    "VHS_SF_CurveToFloat": SF_CurveToFloat,
    "VHS_SF_SpectrumToAmplitude": SF_SpectrumToAmplitude,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SF_AudioToSpectrum": "SF Audio → Spectrum 📊🅥🅗🅢",
    "VHS_SF_SmoothCurve": "SF Smooth Curve 〰️🅥🅗🅢",
    "VHS_SF_WaveGenerator": "SF Wave Generator 🌊🅥🅗🅢",
    "VHS_SF_CurveToFloat": "SF Curve → Float 🔢🅥🅗🅢",
    "VHS_SF_SpectrumToAmplitude": "SF Spectrum → Amplitude 📈🅥🅗🅢",
}
