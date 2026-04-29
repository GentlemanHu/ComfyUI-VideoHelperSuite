"""
DepthFlow Modular Pipeline — Fine-grained ComfyUI nodes for DepthFlow.

Architecture: directly maps to DepthFlow's native DepthState + CudaDepthFlowRenderer.
Existing DepthFlowGenerator node is NOT modified.

Pipeline flow:
  DF Pipeline → DF Set Motion → DF Set State → DF Set PostFX → DF Render
"""
import logging
import math
import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("DepthFlow.Pipeline")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

T_DF_PIPELINE = "DF_PIPELINE"


def _clone_pipe(pipe: Dict) -> Dict:
    return copy.deepcopy(pipe)


def _find_cuda_renderer():
    """Locate CudaDepthFlowRenderer module."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    import importlib.util
    search = [
        Path(__file__).resolve().parent.parent / ".venv_depthflow",
        Path(__file__).resolve().parent.parent.parent.parent / "DepthFlow",
    ]
    for base in search:
        if base is None:
            continue
        candidates = [base / "depthflow" / "cuda_renderer.py"]
        lib_dir = base / "lib"
        if lib_dir.is_dir():
            candidates.extend(lib_dir.glob("python*/site-packages/depthflow/cuda_renderer.py"))
        for c in candidates:
            if c.is_file():
                spec = importlib.util.spec_from_file_location("depthflow.cuda_renderer", str(c))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    try:
        from depthflow import cuda_renderer
        return cuda_renderer
    except ImportError:
        return None


def _estimate_depth(img_np, estimator="da2"):
    """Estimate depth. HF pipeline → luminance fallback."""
    try:
        from transformers import pipeline as hf_pipe
        ids = {"da2": "depth-anything/Depth-Anything-V2-Small-hf",
               "da1": "LiheYoung/depth-anything-small-hf"}
        mid = ids.get(estimator, ids["da2"])
        logger.info(f"Estimating depth with {mid}...")
        from PIL import Image as _PIL
        result = hf_pipe("depth-estimation", model=mid, device=0)(_PIL.fromarray(img_np))
        d = np.array(result["depth"], dtype=np.float32)
        return d / max(d.max(), 1e-6)
    except Exception as e:
        logger.warning(f"HF depth failed ({e}), luminance fallback")
        gray = np.mean(img_np.astype(np.float32), axis=2)
        return 1.0 - (gray / max(gray.max(), 1.0))


# ---------------------------------------------------------------------------
# DF_Pipeline — entry node
# ---------------------------------------------------------------------------

class DF_Pipeline:
    """Create a DepthFlow pipeline with image and optional depth map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "depth_map": ("IMAGE", {"tooltip": "外部深度图（不传则自动估算）"}),
                "depth_estimator": (["da2", "da1", "depthpro", "zoedepth"], {"default": "da2"}),
            },
        }

    RETURN_TYPES = (T_DF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "create"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/DepthFlow Pipeline"

    def create(self, image, depth_map=None, depth_estimator="da2"):
        if HAS_TORCH and isinstance(image, torch.Tensor):
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = np.asarray(image[0])

        # Depth
        depth_np = None
        if depth_map is not None:
            if HAS_TORCH and isinstance(depth_map, torch.Tensor):
                d = depth_map[0].cpu().numpy()
            else:
                d = np.asarray(depth_map[0])
            if d.ndim == 3:
                d = np.mean(d, axis=-1) if d.shape[-1] in (3, 4) else d[:, :, 0]
            depth_np = d.astype(np.float32)
            if depth_np.max() > 1.5:
                depth_np = depth_np / 255.0
        else:
            depth_np = _estimate_depth(img_np, depth_estimator)

        h, w = img_np.shape[:2]
        pipe = {
            "image": img_np,
            "depth": depth_np,
            "width": w, "height": h,
            # Motion defaults
            "motion": {
                "camera_movement": "vertical",
                "intensity": 1.0, "smooth": True, "loop": True,
                "reverse": False, "phase": 0.0,
            },
            # DepthState defaults
            "state": {
                "height": 0.20, "steady": 0.15, "focus": 0.0,
                "zoom": 1.0, "isometric": 0.0, "dolly": 0.0,
                "offset_x": 0.0, "offset_y": 0.0,
                "center_x": 0.0, "center_y": 0.0,
            },
            # PostFX defaults (all off)
            "postfx": {
                "vignette_enable": False, "vignette_intensity": 0.2, "vignette_decay": 20.0,
                "lens_enable": False, "lens_intensity": 0.1, "lens_decay": 0.4,
                "blur_enable": False, "blur_intensity": 1.0,
                "blur_start": 0.6, "blur_end": 1.0,
                "color_saturation": 1.0, "color_contrast": 1.0,
                "color_brightness": 1.0, "color_sepia": 0.0,
            },
        }
        logger.info(f"DF Pipeline created: {w}x{h}, depth={depth_np.shape}")
        return (pipe,)


# ---------------------------------------------------------------------------
# DF_SetMotion — configure camera movement
# ---------------------------------------------------------------------------

class DF_SetMotion:
    """Set camera movement type and parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_DF_PIPELINE,),
                "camera_movement": (["vertical", "horizontal", "zoom",
                                     "circle", "dolly", "orbital", "static"], {
                    "default": "vertical",
                }),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "smooth": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": True}),
                "reverse": ("BOOLEAN", {"default": False}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = (T_DF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "set_motion"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/DepthFlow Pipeline"

    def set_motion(self, pipeline, camera_movement, intensity,
                   smooth=True, loop=True, reverse=False, phase=0.0):
        pipe = _clone_pipe(pipeline)
        pipe["motion"] = {
            "camera_movement": camera_movement,
            "intensity": intensity, "smooth": smooth,
            "loop": loop, "reverse": reverse, "phase": phase,
        }
        logger.info(f"DF Motion: {camera_movement}, intensity={intensity}")
        return (pipe,)


# ---------------------------------------------------------------------------
# DF_SetState — fine-grained depth state control
# ---------------------------------------------------------------------------

class DF_SetState:
    """Fine-grained control over DepthFlow parallax state (maps to DepthState)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_DF_PIPELINE,),
            },
            "optional": {
                "height": ("FLOAT", {"default": 0.20, "min": -2.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "视差强度 (Peak surface height)"}),
                "steady": ("FLOAT", {"default": 0.15, "min": -2.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "焦点深度 (Focal depth for offsets)"}),
                "focus": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "聚焦深度 (Perspective-neutral depth)"}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "isometric": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "等距投影因子 (0=透视 1=正交)"}),
                "dolly": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (T_DF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "set_state"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/DepthFlow Pipeline"

    def set_state(self, pipeline, height=0.20, steady=0.15, focus=0.0,
                  zoom=1.0, isometric=0.0, dolly=0.0,
                  offset_x=0.0, offset_y=0.0):
        pipe = _clone_pipe(pipeline)
        pipe["state"] = {
            "height": height, "steady": steady, "focus": focus,
            "zoom": zoom, "isometric": isometric, "dolly": dolly,
            "offset_x": offset_x, "offset_y": offset_y,
            "center_x": 0.0, "center_y": 0.0,
        }
        logger.info(f"DF State: height={height}, steady={steady}, iso={isometric}")
        return (pipe,)


# ---------------------------------------------------------------------------
# DF_SetPostFX — post-processing effects
# ---------------------------------------------------------------------------

class DF_SetPostFX:
    """Configure DepthFlow post-processing: vignette, lens blur, depth blur, color."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_DF_PIPELINE,),
            },
            "optional": {
                "vignette_enable": ("BOOLEAN", {"default": False}),
                "vignette_intensity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vignette_decay": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "lens_enable": ("BOOLEAN", {"default": False}),
                "lens_intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blur_enable": ("BOOLEAN", {"default": False}),
                "blur_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blur_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "景深模糊起始深度"}),
                "blur_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_sepia": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (T_DF_PIPELINE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "set_postfx"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/DepthFlow Pipeline"

    def set_postfx(self, pipeline,
                   vignette_enable=False, vignette_intensity=0.2, vignette_decay=20.0,
                   lens_enable=False, lens_intensity=0.1,
                   blur_enable=False, blur_intensity=1.0, blur_start=0.6, blur_end=1.0,
                   color_saturation=1.0, color_contrast=1.0,
                   color_brightness=1.0, color_sepia=0.0):
        pipe = _clone_pipe(pipeline)
        pipe["postfx"] = {
            "vignette_enable": vignette_enable,
            "vignette_intensity": vignette_intensity,
            "vignette_decay": vignette_decay,
            "lens_enable": lens_enable,
            "lens_intensity": lens_intensity,
            "lens_decay": 0.4,
            "blur_enable": blur_enable,
            "blur_intensity": blur_intensity,
            "blur_start": blur_start,
            "blur_end": blur_end,
            "color_saturation": color_saturation,
            "color_contrast": color_contrast,
            "color_brightness": color_brightness,
            "color_sepia": color_sepia,
        }
        logger.info(f"DF PostFX: vig={vignette_enable}, lens={lens_enable}, blur={blur_enable}")
        return (pipe,)


# ---------------------------------------------------------------------------
# DF_Render — O(1) memory streaming render
# ---------------------------------------------------------------------------

class DF_Render:
    """Render the DepthFlow pipeline to video. Uses CUDA render_frame() + ffmpeg
    pipe for O(1) memory. Falls back to render_video() if needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (T_DF_PIPELINE,),
            },
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                  "tooltip": "0=使用原图尺寸"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "ssaa": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 2.0, "step": 0.25}),
                "quality": ("INT", {"default": 96, "min": 1, "max": 100}),
                "codec": (["h264", "h264-nvenc", "h265", "h265-nvenc"], {"default": "h264"}),
                "output_format": (["mp4", "mkv", "webm"], {"default": "mp4"}),
                "output_prefix": ("STRING", {"default": "df_pipe_"}),
                "output_frames": ("BOOLEAN", {"default": False,
                                              "tooltip": "同时输出 IMAGE tensor（占内存）"}),
                "enable_inpaint": ("BOOLEAN", {"default": False}),
                "inpaint_threshold": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 8.0, "step": 0.1}),
                "enable_aa": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/DepthFlow Pipeline"

    def render(self, pipeline, width=0, height=0, fps=30, duration=5.0,
               ssaa=1.0, quality=80, codec="h264", output_format="mp4",
               output_prefix="df_pipe_", output_frames=False,
               enable_inpaint=False, inpaint_threshold=2.2, enable_aa=True):
        import folder_paths
        from datetime import datetime
        from comfy.utils import ProgressBar
        import subprocess, shutil

        pipe = pipeline
        img_np = pipe["image"]
        depth_np = pipe["depth"]
        motion = pipe["motion"]
        state_cfg = pipe["state"]
        postfx = pipe["postfx"]

        # Resolve dimensions
        rw = width if width > 0 else pipe["width"]
        rh = height if height > 0 else pipe["height"]
        # Ensure even
        rw = rw - (rw % 2)
        rh = rh - (rh % 2)

        total_frames = max(1, int(duration * fps))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(folder_paths.get_output_directory(), "depthflow_videos")
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{output_prefix}{stamp}_{rw}x{rh}.{output_format}"
        out_path = os.path.join(out_dir, out_name)

        logger.info(f"DF Render: {rw}x{rh} @ {fps}fps, {duration}s, {total_frames} frames")

        # Try CUDA path
        cuda_mod = _find_cuda_renderer()
        if cuda_mod is not None and cuda_mod.is_available():
            logger.info("DF Render: CUDA path")
            frames_tensor = self._render_cuda(
                cuda_mod, img_np, depth_np, rw, rh, fps, duration, ssaa,
                quality, codec, output_format, out_path, total_frames,
                motion, state_cfg, postfx, enable_inpaint, inpaint_threshold, enable_aa,
                output_frames,
            )
        else:
            # Fallback: use render_video via existing DepthFlowGenerator
            logger.info("DF Render: fallback to subprocess")
            frames_tensor = self._render_subprocess(
                img_np, depth_np, rw, rh, fps, duration, ssaa,
                quality, codec, output_format, out_path, total_frames,
                motion,
            )

        final_path = os.path.abspath(out_path)

        if frames_tensor is None:
            frames_tensor = torch.zeros((1, rh, rw, 3), dtype=torch.float32)

        return {
            "ui": {"video": [{"filename": out_name, "subfolder": "depthflow_videos", "type": "output"}]},
            "result": (final_path, frames_tensor),
        }

    def _render_cuda(self, cuda_mod, img_np, depth_np, rw, rh, fps, duration,
                     ssaa, quality, codec, fmt, out_path, total_frames,
                     motion, state_cfg, postfx,
                     enable_inpaint, inpaint_threshold, enable_aa, output_frames):
        """O(1) memory: render_frame() per frame → ffmpeg pipe."""
        from comfy.utils import ProgressBar
        import subprocess, shutil, time

        img_f = img_np.astype(np.float32) / 255.0
        dep_f = depth_np.astype(np.float32)
        renderer = cuda_mod.CudaDepthFlowRenderer(img_f, dep_f)

        ssaa_w = int(rw * ssaa)
        ssaa_h = int(rh * ssaa)

        # ffmpeg command
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        vcodec_map = {"h264": "libx264", "h264-nvenc": "h264_nvenc",
                      "h265": "libx265", "h265-nvenc": "hevc_nvenc"}
        vcodec = vcodec_map.get(codec, "libx264")

        cmd = [ffmpeg_bin, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{ssaa_w}x{ssaa_h}", "-r", str(fps), "-i", "pipe:0"]
        if ssaa != 1.0:
            cmd += ["-vf", f"scale={rw}:{rh}:flags=lanczos"]
        cmd += ["-c:v", vcodec, "-pix_fmt", "yuv420p", "-an"]
        if "nvenc" in vcodec:
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "18"]
        else:
            cmd += ["-preset", "fast", "-crf", "18"]
        cmd.append(out_path)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        pbar = ProgressBar(total_frames)
        captured = []
        t0 = time.perf_counter()

        try:
            for fi in range(total_frames):
                tau = fi / max(total_frames - 1, 1)
                # Compute animation state from motion params
                state = cuda_mod.compute_animation_state(
                    motion["camera_movement"], tau,
                    intensity=motion["intensity"],
                    smooth=motion["smooth"],
                    loop=motion["loop"],
                    reverse=motion["reverse"],
                    phase=motion["phase"],
                    steady_depth=state_cfg.get("steady", 0.15),
                    isometric=state_cfg.get("isometric", 0.0),
                )
                # Override with user state settings
                state.height = state_cfg.get("height", state.height)
                state.focus = state_cfg.get("focus", state.focus)
                state.zoom = state_cfg.get("zoom", state.zoom)
                state.dolly = state_cfg.get("dolly", state.dolly)

                # Apply PostFX to state
                pfx = postfx
                if pfx.get("vignette_enable"):
                    state.vig_enable = True
                    state.vig_intensity = pfx["vignette_intensity"]
                    state.vig_decay = pfx["vignette_decay"]
                if pfx.get("lens_enable"):
                    state.lens_enable = True
                    state.lens_intensity = pfx["lens_intensity"]
                if pfx.get("blur_enable"):
                    state.blur_enable = True
                    state.blur_intensity = pfx["blur_intensity"]
                    state.blur_start = pfx["blur_start"]
                    state.blur_end = pfx["blur_end"]
                state.color_saturation = pfx.get("color_saturation", 1.0)
                state.color_contrast = pfx.get("color_contrast", 1.0)
                state.color_brightness = pfx.get("color_brightness", 1.0)
                state.color_sepia = pfx.get("color_sepia", 0.0)

                eff_aa = enable_aa and (ssaa <= 1.0)
                frame = renderer.render_frame(
                    ssaa_w, ssaa_h, state, quality,
                    enable_inpaint=enable_inpaint,
                    inpaint_threshold=inpaint_threshold,
                    inpaint_mode="soften" if enable_inpaint else "off",
                    enable_aa=eff_aa,
                )
                proc.stdin.write(frame.numpy().tobytes())

                if output_frames:
                    captured.append(frame)

                pbar.update_absolute(fi + 1, total_frames)
                if (fi + 1) % 60 == 0 or fi == total_frames - 1:
                    elapsed = time.perf_counter() - t0
                    r_fps = (fi + 1) / max(elapsed, 0.001)
                    logger.info(f"DF Render: {fi+1}/{total_frames} ({r_fps:.1f} fps)")
        finally:
            proc.stdin.close()
            proc.wait()

        elapsed = time.perf_counter() - t0
        logger.info(f"DF Render: done in {elapsed:.1f}s")

        if output_frames and captured:
            return torch.stack(captured).float() / 255.0
        return None

    def _render_subprocess(self, img_np, depth_np, rw, rh, fps, duration,
                           ssaa, quality, codec, fmt, out_path, total_frames,
                           motion):
        """Fallback: use CudaDepthFlowRenderer.render_video() directly."""
        cuda_mod = _find_cuda_renderer()
        if cuda_mod is None:
            logger.error("DF Render: no CUDA renderer available for fallback")
            return None

        img_f = img_np.astype(np.float32) / 255.0
        dep_f = depth_np.astype(np.float32)
        renderer = cuda_mod.CudaDepthFlowRenderer(img_f, dep_f)
        from comfy.utils import ProgressBar
        pbar = ProgressBar(total_frames)

        renderer.render_video(
            output_path=out_path, render_w=rw, render_h=rh,
            fps=fps, duration=duration, ssaa=ssaa, quality_pct=quality,
            camera_movement=motion["camera_movement"],
            intensity=motion["intensity"], smooth=motion["smooth"],
            loop=motion["loop"], reverse=motion["reverse"],
            phase=motion["phase"],
            codec=codec, output_format=fmt,
            progress_cb=lambda c, t: pbar.update_absolute(c, t),
        )
        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VHS_DF_Pipeline": DF_Pipeline,
    "VHS_DF_SetMotion": DF_SetMotion,
    "VHS_DF_SetState": DF_SetState,
    "VHS_DF_SetPostFX": DF_SetPostFX,
    "VHS_DF_Render": DF_Render,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_DF_Pipeline": "DF Pipeline 🌊🅥🅗🅢",
    "VHS_DF_SetMotion": "DF Set Motion 🎬🅥🅗🅢",
    "VHS_DF_SetState": "DF Set State ⚙️🅥🅗🅢",
    "VHS_DF_SetPostFX": "DF Set PostFX ✨🅥🅗🅢",
    "VHS_DF_Render": "DF Render 🎥🅥🅗🅢",
}
