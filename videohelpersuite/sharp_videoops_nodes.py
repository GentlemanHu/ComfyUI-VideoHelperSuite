"""Apple SHARP video nodes for VideoHelperSuite."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

import folder_paths
from comfy.utils import ProgressBar

from . import sharp_engine
from .utils import ffmpeg_path


T_SHARP_SCENE = "VHS_SHARP_SCENE"
T_SHARP_CAMERA = "VHS_SHARP_CAMERA"

CAMERA_PRESETS = [
    "official_rotate_forward",
    "official_rotate",
    "official_swipe",
    "official_shake",
    "viewer_auto",
    "cinematic_orbit",
    "portrait_orbit",
    "portrait_push",
    "window_parallax",
    "slow_gallery",
    "breathing_closeup",
    "dolly_push",
    "dolly_pull",
    "crane_up",
    "truck_left",
    "arc_reveal",
    "hero_parallax",
    "micro_float",
    "turntable",
    "custom",
]

PERFORMANCE_PRESETS = ["viewer", "draft", "balanced", "quality", "custom"]
RESOLUTION_MODES = ["auto_source", "custom", "720p", "1080p", "square_1024", "source"]
SPLAT_QUALITIES = ["point", "fast", "balanced"]
RENDER_BACKENDS = ["auto", "gpu", "cpu"]
RENDER_MODES = ["photo_composite", "source_static", "gaussian_color", "depth", "alpha"]


def _source_size_from_image(image) -> tuple[int, int]:
    if isinstance(image, torch.Tensor):
        data = image
        if data.dim() == 4:
            return int(data.shape[2]), int(data.shape[1])
        if data.dim() == 3:
            return int(data.shape[1]), int(data.shape[0])
    return 1024, 1024


def _source_size_from_scene(scene: sharp_engine.SharpScene) -> tuple[int, int]:
    return int(scene.source_size[0]), int(scene.source_size[1])


def _even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def _resolve_size(mode: str, width: int, height: int, source_size: tuple[int, int]) -> tuple[int, int]:
    src_w, src_h = max(1, int(source_size[0])), max(1, int(source_size[1]))
    aspect = src_w / src_h
    key = str(mode or "custom")
    if key == "source":
        return _even(src_w), _even(src_h)
    if key == "720p":
        return (_even(1280), _even(720)) if aspect >= 1 else (_even(720), _even(1280))
    if key == "1080p":
        return (_even(1920), _even(1080)) if aspect >= 1 else (_even(1080), _even(1920))
    if key == "square_1024":
        return 1024, 1024
    if key == "auto_source":
        max_side = 1024
        if src_w >= src_h:
            return _even(max_side), _even(round(max_side / aspect))
        return _even(round(max_side * aspect)), _even(max_side)
    return _even(width), _even(height)


def _resolve_budget(mode: str, max_gaussians: int) -> int:
    key = str(mode or "balanced")
    if key == "viewer":
        return min(int(max_gaussians), 80000) if max_gaussians > 0 else 80000
    if key == "draft":
        return min(int(max_gaussians), 40000) if max_gaussians > 0 else 40000
    if key == "balanced":
        return min(int(max_gaussians), 100000) if max_gaussians > 0 else 100000
    if key == "quality":
        return min(int(max_gaussians), 250000) if max_gaussians > 0 else 250000
    return int(max_gaussians)


def _bg(color: str) -> tuple[float, float, float]:
    c = str(color).strip().lower()
    table = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "transparent_black": (0.0, 0.0, 0.0),
    }
    if c in table:
        return table[c]
    if c.startswith("#") and len(c) == 7:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return (0.0, 0.0, 0.0)


def _empty_frames(width: int, height: int) -> torch.Tensor:
    return torch.zeros((1, int(height), int(width), 3), dtype=torch.float32)


def _encode_video(frames: list[torch.Tensor], fps: int, codec: str, output_prefix: str) -> tuple[str, str]:
    sharp_engine.log_info(f"FFmpeg encode start: frames={len(frames)}, fps={fps}, codec={codec}, prefix={output_prefix}")
    t0 = time.perf_counter()
    out_dir = Path(folder_paths.get_output_directory()) / "sharp_videoops"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{output_prefix}_{int(time.time() * 1000)}.mp4"
    out_path = out_dir / filename
    h, w = frames[0].shape[:2]
    ffmpeg_bin = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
    codec_map = {
        "h264": "libx264",
        "h265": "libx265",
        "h264-nvenc": "h264_nvenc",
        "h265-nvenc": "hevc_nvenc",
    }
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-c:v",
        codec_map.get(codec, "libx264"),
        "-pix_fmt",
        "yuv420p",
    ]
    if "nvenc" in codec:
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "18"]
    else:
        cmd += ["-preset", "fast", "-crf", "18"]
    cmd += ["-an", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            proc.stdin.write(arr.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"SHARP video ffmpeg encode failed with code {proc.returncode}")
    size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0.0
    sharp_engine.log_info(f"FFmpeg encode complete in {time.perf_counter() - t0:.2f}s: {out_path} ({size_mb:.2f} MB)")
    return str(out_path), filename


def _render_frames(
    scene: sharp_engine.SharpScene,
    camera: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: int,
    duration: float,
    splat_size: float,
    opacity_gain: float,
    exposure: float,
    gamma: float,
    background: str,
    render_backend: str,
    splat_quality: str,
    render_mode: str,
    source_photo_strength: float,
) -> list[torch.Tensor]:
    total = max(1, int(round(float(duration) * int(fps))))
    device = sharp_engine.render_device(render_backend)
    gaussian_count = int(scene.gaussians.mean_vectors.reshape(-1, 3).shape[0])
    megapixels = (int(width) * int(height)) / 1_000_000.0
    sharp_engine.log_info(
        f"Render frames start: size={width}x{height}, fps={fps}, duration={duration}, "
        f"frames={total}, gaussians={gaussian_count}, megapixels={megapixels:.2f}, "
        f"backend={render_backend}, device={device}, splat_quality={splat_quality}, "
        f"mode={render_mode}"
    )
    t0 = time.perf_counter()
    pbar = ProgressBar(total)
    frames: list[torch.Tensor] = []
    for i in range(total):
        frame_t0 = time.perf_counter()
        tau = i / max(total - 1, 1)
        if i == 0:
            sharp_engine.log_info("Render first frame start: gsplat may compile CUDA kernels on first use")
        frame = sharp_engine.render_frame(
            scene,
            camera,
            tau,
            int(width),
            int(height),
            splat_size=float(splat_size),
            opacity_gain=float(opacity_gain),
            exposure=float(exposure),
            gamma=float(gamma),
            background=_bg(background),
            render_backend=render_backend,
            splat_quality=splat_quality,
            render_mode=render_mode,
            source_photo_strength=float(source_photo_strength),
        )
        frames.append(frame)
        pbar.update_absolute(i + 1, total)
        if i == 0 or (i + 1) % max(1, total // 8) == 0 or i == total - 1:
            sharp_engine.log_info(
                f"Render progress: {i + 1}/{total} frames, "
                f"last_frame={time.perf_counter() - frame_t0:.3f}s"
            )
    sharp_engine.log_info(f"Render frames complete in {time.perf_counter() - t0:.2f}s")
    return frames


def _scene_info(scene: sharp_engine.SharpScene) -> str:
    g = scene.gaussians
    return json.dumps(
        {
            "engine": "VHS independent Apple SHARP",
            "model": "apple/Sharp",
            "gaussians": int(g.mean_vectors.reshape(-1, 3).shape[0]),
            "source_size": list(scene.source_size),
            "focal_px": scene.focal_px,
            "radius": scene.radius,
            "ply_path": scene.ply_path,
        },
        ensure_ascii=False,
        indent=2,
    )


class VHSSharpBuildScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "focal_length_mm": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "max_gaussians": ("INT", {"default": 120000, "min": 0, "max": 1000000, "step": 1000}),
                "min_opacity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "sharp_scene"}),
                "performance_preset": (PERFORMANCE_PRESETS, {"default": "viewer"}),
            },
        }

    RETURN_TYPES = (T_SHARP_SCENE, "STRING", "STRING")
    RETURN_NAMES = ("scene", "ply_path", "info")
    FUNCTION = "build"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def build(
        self,
        image,
        precision="auto",
        focal_length_mm=30.0,
        max_gaussians=120000,
        min_opacity=0.02,
        save_ply=True,
        output_prefix="sharp_scene",
        performance_preset="viewer",
    ):
        budget = _resolve_budget(performance_preset, int(max_gaussians))
        sharp_engine.log_info(
            f"Node BuildScene start: preset={performance_preset}, gaussian_budget={budget}, save_ply={save_ply}"
        )
        scene = sharp_engine.make_scene(
            image,
            precision=precision,
            focal_length_mm=float(focal_length_mm),
            max_gaussians=budget,
            min_opacity=float(min_opacity),
            save_ply=bool(save_ply),
            output_prefix=str(output_prefix),
        )
        return (scene, scene.ply_path, _scene_info(scene))


class VHSSharpCameraRig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (CAMERA_PRESETS, {"default": "official_rotate_forward"}),
            },
            "optional": {
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "radius_scale": ("FLOAT", {"default": 1.35, "min": 0.2, "max": 5.0, "step": 0.01}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -80.0, "max": 80.0, "step": 1.0}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "yaw_delta": ("FLOAT", {"default": 20.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "pitch_delta": ("FLOAT", {"default": 0.0, "min": -160.0, "max": 160.0, "step": 1.0}),
                "radius_delta": ("FLOAT", {"default": 0.0, "min": -0.9, "max": 2.0, "step": 0.01}),
                "roll_delta": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = (T_SHARP_CAMERA, "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "build"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def build(
        self,
        preset,
        amplitude=1.0,
        radius_scale=1.35,
        yaw=0.0,
        pitch=0.0,
        phase=0.0,
        yaw_delta=20.0,
        pitch_delta=0.0,
        radius_delta=0.0,
        roll_delta=0.0,
    ):
        camera = {
            "preset": preset,
            "amplitude": float(amplitude),
            "radius_scale": float(radius_scale),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "phase": float(phase),
            "yaw_delta": float(yaw_delta),
            "pitch_delta": float(pitch_delta),
            "radius_delta": float(radius_delta),
            "roll_delta": float(roll_delta),
        }
        return (camera, json.dumps(camera, ensure_ascii=False, indent=2))


class VHSSharpRenderVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": (T_SHARP_SCENE,),
                "camera": (T_SHARP_CAMERA,),
            },
            "optional": {
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 720, "min": 256, "max": 4096, "step": 8}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "duration": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "splat_size": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 8.0, "step": 0.05}),
                "opacity_gain": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 8.0, "step": 0.05}),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05}),
                "background": (["black", "white", "gray", "transparent_black", "#101014"], {"default": "black"}),
                "video_codec": (["h264", "h265", "h264-nvenc", "h265-nvenc"], {"default": "h264"}),
                "output_prefix": ("STRING", {"default": "sharp_video"}),
                "output_frames": ("BOOLEAN", {"default": True}),
                "resolution_mode": (RESOLUTION_MODES, {"default": "custom"}),
                "render_backend": (RENDER_BACKENDS, {"default": "auto"}),
                "splat_quality": (SPLAT_QUALITIES, {"default": "balanced"}),
                "render_mode": (RENDER_MODES, {"default": "photo_composite"}),
                "source_photo_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "frames", "frame_count", "duration")
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def render(
        self,
        scene,
        camera,
        width=1280,
        height=720,
        fps=24,
        duration=4.0,
        splat_size=1.0,
        opacity_gain=1.25,
        exposure=1.0,
        gamma=1.0,
        background="black",
        video_codec="h264",
        output_prefix="sharp_video",
        output_frames=True,
        resolution_mode="custom",
        render_backend="auto",
        splat_quality="balanced",
        render_mode="photo_composite",
        source_photo_strength=0.85,
    ):
        out_w, out_h = _resolve_size(resolution_mode, int(width), int(height), _source_size_from_scene(scene))
        sharp_engine.log_info(
            f"Node RenderVideo start: camera={camera.get('name', 'custom')}, "
            f"resolution_mode={resolution_mode}, output={out_w}x{out_h}, codec={video_codec}"
        )
        frames = _render_frames(
            scene,
            camera,
            width=out_w,
            height=out_h,
            fps=int(fps),
            duration=float(duration),
            splat_size=float(splat_size),
            opacity_gain=float(opacity_gain),
            exposure=float(exposure),
            gamma=float(gamma),
            background=str(background),
            render_backend=str(render_backend),
            splat_quality=str(splat_quality),
            render_mode=str(render_mode),
            source_photo_strength=float(source_photo_strength),
        )
        video_path, filename = _encode_video(frames, int(fps), str(video_codec), str(output_prefix))
        frames_tensor = torch.stack(frames, dim=0) if output_frames else _empty_frames(out_w, out_h)
        return {
            "ui": {"video": [{"filename": filename, "subfolder": "sharp_videoops", "type": "output"}]},
            "result": (os.path.abspath(video_path), frames_tensor, len(frames), float(duration)),
        }


class VHSSharpImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "camera_preset": (CAMERA_PRESETS, {"default": "official_rotate_forward"}),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "focal_length_mm": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "max_gaussians": ("INT", {"default": 120000, "min": 0, "max": 1000000, "step": 1000}),
                "min_opacity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 720, "min": 256, "max": 4096, "step": 8}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "duration": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "motion_amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "radius_scale": ("FLOAT", {"default": 1.35, "min": 0.2, "max": 5.0, "step": 0.01}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -80.0, "max": 80.0, "step": 1.0}),
                "splat_size": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 8.0, "step": 0.05}),
                "opacity_gain": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 8.0, "step": 0.05}),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05}),
                "background": (["black", "white", "gray", "transparent_black", "#101014"], {"default": "black"}),
                "video_codec": (["h264", "h265", "h264-nvenc", "h265-nvenc"], {"default": "h264"}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "sharp_image_to_video"}),
                "output_frames": ("BOOLEAN", {"default": True}),
                "performance_preset": (PERFORMANCE_PRESETS, {"default": "viewer"}),
                "resolution_mode": (RESOLUTION_MODES, {"default": "auto_source"}),
                "render_backend": (RENDER_BACKENDS, {"default": "auto"}),
                "splat_quality": (SPLAT_QUALITIES, {"default": "balanced"}),
                "render_mode": (RENDER_MODES, {"default": "photo_composite"}),
                "source_photo_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", T_SHARP_SCENE, "STRING", "STRING")
    RETURN_NAMES = ("video_path", "frames", "scene", "ply_path", "info")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def run(self, image, camera_preset, **kwargs):
        budget = _resolve_budget(kwargs.get("performance_preset", "balanced"), int(kwargs.get("max_gaussians", 120000)))
        source_size = _source_size_from_image(image)
        out_w, out_h = _resolve_size(
            kwargs.get("resolution_mode", "auto_source"),
            int(kwargs.get("width", 1280)),
            int(kwargs.get("height", 720)),
            source_size,
        )
        sharp_engine.log_info(
            f"Node ImageToVideo start: camera={camera_preset}, source={source_size[0]}x{source_size[1]}, "
            f"output={out_w}x{out_h}, resolution_mode={kwargs.get('resolution_mode', 'auto_source')}, "
            f"preset={kwargs.get('performance_preset', 'balanced')}, gaussian_budget={budget}, "
            f"render_mode={kwargs.get('render_mode', 'photo_composite')}, codec={kwargs.get('video_codec', 'h264')}"
        )
        scene = sharp_engine.make_scene(
            image,
            precision=kwargs.get("precision", "auto"),
            focal_length_mm=float(kwargs.get("focal_length_mm", 30.0)),
            max_gaussians=budget,
            min_opacity=float(kwargs.get("min_opacity", 0.02)),
            save_ply=bool(kwargs.get("save_ply", True)),
            output_prefix=str(kwargs.get("output_prefix", "sharp_image_to_video")),
        )
        camera = {
            "preset": camera_preset,
            "amplitude": float(kwargs.get("motion_amplitude", 1.0)),
            "radius_scale": float(kwargs.get("radius_scale", 1.35)),
            "yaw": float(kwargs.get("yaw", 0.0)),
            "pitch": float(kwargs.get("pitch", 0.0)),
            "phase": 0.0,
        }
        frames = _render_frames(
            scene,
            camera,
            width=out_w,
            height=out_h,
            fps=int(kwargs.get("fps", 24)),
            duration=float(kwargs.get("duration", 4.0)),
            splat_size=float(kwargs.get("splat_size", 1.0)),
            opacity_gain=float(kwargs.get("opacity_gain", 1.25)),
            exposure=float(kwargs.get("exposure", 1.0)),
            gamma=float(kwargs.get("gamma", 1.0)),
            background=str(kwargs.get("background", "black")),
            render_backend=str(kwargs.get("render_backend", "auto")),
            splat_quality=str(kwargs.get("splat_quality", "balanced")),
            render_mode=str(kwargs.get("render_mode", "photo_composite")),
            source_photo_strength=float(kwargs.get("source_photo_strength", 0.85)),
        )
        video_path, filename = _encode_video(
            frames,
            int(kwargs.get("fps", 24)),
            str(kwargs.get("video_codec", "h264")),
            str(kwargs.get("output_prefix", "sharp_image_to_video")),
        )
        frames_tensor = torch.stack(frames, dim=0) if kwargs.get("output_frames", True) else _empty_frames(out_w, out_h)
        info = _scene_info(scene)
        return {
            "ui": {"video": [{"filename": filename, "subfolder": "sharp_videoops", "type": "output"}]},
            "result": (os.path.abspath(video_path), frames_tensor, scene, scene.ply_path, info),
        }


NODE_CLASS_MAPPINGS = {
    "VHS_SHARP_BuildScene": VHSSharpBuildScene,
    "VHS_SHARP_CameraRig": VHSSharpCameraRig,
    "VHS_SHARP_RenderVideo": VHSSharpRenderVideo,
    "VHS_SHARP_ImageToVideo": VHSSharpImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SHARP_BuildScene": "SHARP Build Gaussian Scene 🍎🎥🅥🅗🅢",
    "VHS_SHARP_CameraRig": "SHARP Camera Rig 🍎🎬🅥🅗🅢",
    "VHS_SHARP_RenderVideo": "SHARP Render Video 🍎🎥🅥🅗🅢",
    "VHS_SHARP_ImageToVideo": "SHARP Image To Video 🍎🎥🅥🅗🅢",
}
