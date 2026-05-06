"""Extra SHARP VideoOps nodes for PLY reuse, presets, and reference motion."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import folder_paths

from . import sharp_engine
from .depthflow_adapter import estimate_depth, prepare_depth_map
from .sharp_scene_ops import filter_scene, load_ply_scene, merge_scenes, save_scene_ply, scene_info, transform_scene
from .sharp_videoops_nodes import DEFAULT_CAMERA_PATH, T_SHARP_CAMERA, T_SHARP_SCENE, _parse_camera_path


PRESET_DIR = Path(folder_paths.get_output_directory()) / "sharp_videoops" / "presets"


def _image_tensor_to_np(image: torch.Tensor, index: int = 0) -> np.ndarray:
    data = image[index] if image.dim() == 4 else image
    return (data.detach().cpu().numpy()[..., :3] * 255.0).clip(0, 255).astype(np.uint8)


def _depth_to_tensor(depth: np.ndarray) -> torch.Tensor:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = np.repeat(depth[..., None], 3, axis=2)
    return torch.from_numpy(depth[None]).float().clamp(0, 1)


def _preset_path(name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    if not safe:
        raise ValueError("SHARP preset name cannot be empty")
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    return PRESET_DIR / f"{safe}.json"


def _camera_info(camera: dict[str, Any]) -> str:
    return json.dumps(camera, ensure_ascii=False, indent=2)


def _estimate_reference_motion(frames: torch.Tensor, samples: int, strength: float) -> list[dict[str, float]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("SHARP reference motion extraction requires opencv-python/cv2") from exc
    if not isinstance(frames, torch.Tensor) or frames.dim() != 4 or frames.shape[0] < 2:
        raise ValueError("Reference video motion needs IMAGE batch with at least two frames")
    total = int(frames.shape[0])
    count = max(2, min(int(samples), total))
    indices = np.linspace(0, total - 1, count).round().astype(int).tolist()
    first = cv2.cvtColor(_image_tensor_to_np(frames, 0), cv2.COLOR_RGB2GRAY)
    features0 = cv2.goodFeaturesToTrack(first, maxCorners=900, qualityLevel=0.01, minDistance=8)
    keyframes: list[dict[str, float]] = []
    for idx in indices:
        current = cv2.cvtColor(_image_tensor_to_np(frames, idx), cv2.COLOR_RGB2GRAY)
        dx = dy = 0.0
        scale = 1.0
        if features0 is not None and len(features0) >= 8:
            pts1, status, _ = cv2.calcOpticalFlowPyrLK(first, current, features0, None)
            valid = status.reshape(-1) > 0 if status is not None else np.zeros((0,), dtype=bool)
            if pts1 is not None and int(valid.sum()) >= 8:
                affine, _ = cv2.estimateAffinePartial2D(features0[valid], pts1[valid], method=cv2.RANSAC)
                if affine is not None:
                    dx = float(affine[0, 2])
                    dy = float(affine[1, 2])
                    scale = float((affine[0, 0] ** 2 + affine[1, 0] ** 2) ** 0.5)
        width = max(1, int(frames.shape[2]))
        height = max(1, int(frames.shape[1]))
        keyframes.append(
            {
                "t": float(idx / max(total - 1, 1)),
                "x": float(np.clip(-dx / width * 3.0 * strength, -2.0, 2.0)),
                "y": float(np.clip(dy / height * 3.0 * strength, -2.0, 2.0)),
                "z": float(np.clip((1.0 - scale) * 4.0 * strength, -2.0, 2.0)),
            }
        )
    keyframes[0]["t"] = 0.0
    keyframes[-1]["t"] = 1.0
    return keyframes


class VHSSharpLoadPlyScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ply_path": ("STRING", {"default": ""})},
            "optional": {
                "source_image": ("IMAGE",),
                "focal_px_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "max_gaussians": ("INT", {"default": 0, "min": 0, "max": 2000000, "step": 1000}),
                "min_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (T_SHARP_SCENE, "STRING")
    RETURN_NAMES = ("scene", "info")
    FUNCTION = "load"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def load(self, ply_path, source_image=None, focal_px_override=0.0, max_gaussians=0, min_opacity=0.0):
        sharp_engine.log_info(f"Node LoadPlyScene start: path={ply_path}")
        scene = load_ply_scene(str(ply_path), source_image=source_image, focal_px=float(focal_px_override))
        if int(max_gaussians) > 0 or float(min_opacity) > 0:
            scene = filter_scene(scene, int(max_gaussians), float(min_opacity))
        return (scene, scene_info(scene))


class VHSSharpSceneProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"scene": (T_SHARP_SCENE,)},
            "optional": {
                "max_gaussians": ("INT", {"default": 0, "min": 0, "max": 2000000, "step": 1000}),
                "min_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "translate_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "rotate_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "save_ply": ("BOOLEAN", {"default": False}),
                "output_prefix": ("STRING", {"default": "sharp_scene_processed"}),
            },
        }

    RETURN_TYPES = (T_SHARP_SCENE, "STRING", "STRING")
    RETURN_NAMES = ("scene", "ply_path", "info")
    FUNCTION = "process"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def process(self, scene, **kwargs):
        sharp_engine.log_info("Node SceneProcess start")
        out = scene
        if int(kwargs.get("max_gaussians", 0)) > 0 or float(kwargs.get("min_opacity", 0.0)) > 0:
            out = filter_scene(out, int(kwargs.get("max_gaussians", 0)), float(kwargs.get("min_opacity", 0.0)))
        out = transform_scene(
            out,
            (float(kwargs.get("translate_x", 0.0)), float(kwargs.get("translate_y", 0.0)), float(kwargs.get("translate_z", 0.0))),
            float(kwargs.get("scale", 1.0)),
            float(kwargs.get("rotate_y", 0.0)),
        )
        ply_path = save_scene_ply(out, str(kwargs.get("output_prefix", "sharp_scene_processed"))) if kwargs.get("save_ply", False) else ""
        return (out, ply_path, scene_info(out))


class VHSSharpMergeScenes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"scene_a": (T_SHARP_SCENE,), "scene_b": (T_SHARP_SCENE,)},
            "optional": {
                "scene_c": (T_SHARP_SCENE,),
                "max_gaussians": ("INT", {"default": 0, "min": 0, "max": 3000000, "step": 1000}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "sharp_scene_merged"}),
            },
        }

    RETURN_TYPES = (T_SHARP_SCENE, "STRING", "STRING")
    RETURN_NAMES = ("scene", "ply_path", "info")
    FUNCTION = "merge"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def merge(self, scene_a, scene_b, scene_c=None, max_gaussians=0, save_ply=True, output_prefix="sharp_scene_merged"):
        scenes = [scene_a, scene_b] + ([scene_c] if scene_c is not None else [])
        scene = merge_scenes(scenes, int(max_gaussians))
        ply_path = save_scene_ply(scene, str(output_prefix)) if save_ply else ""
        return (scene, ply_path, scene_info(scene))


class VHSSharpMergePlyFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ply_folder": ("STRING", {"default": ""})},
            "optional": {
                "source_image": ("IMAGE",),
                "pattern": ("STRING", {"default": "*.ply"}),
                "max_files": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "max_gaussians": ("INT", {"default": 250000, "min": 0, "max": 5000000, "step": 1000}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "sharp_ply_folder_merged"}),
            },
        }

    RETURN_TYPES = (T_SHARP_SCENE, "STRING", "STRING")
    RETURN_NAMES = ("scene", "ply_path", "info")
    FUNCTION = "merge"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def merge(self, ply_folder, source_image=None, pattern="*.ply", max_files=32, max_gaussians=250000, save_ply=True, output_prefix="sharp_ply_folder_merged"):
        folder = Path(str(ply_folder)).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"SHARP PLY folder not found: {folder}")
        files = sorted(folder.glob(str(pattern)))[: int(max_files)]
        if not files:
            raise FileNotFoundError(f"No PLY files matched {pattern} in {folder}")
        sharp_engine.log_info(f"Node MergePlyFolder start: files={len(files)}, max_gaussians={max_gaussians}")
        scenes = [load_ply_scene(str(path), source_image=source_image) for path in files]
        scene = merge_scenes(scenes, int(max_gaussians))
        ply_path = save_scene_ply(scene, str(output_prefix)) if save_ply else ""
        return (scene, ply_path, scene_info(scene))


class VHSSharpSaveCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"camera": (T_SHARP_CAMERA,), "preset_name": ("STRING", {"default": "my_sharp_camera"})},
            "optional": {"overwrite": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("preset_path", "info")
    FUNCTION = "save"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def save(self, camera, preset_name, overwrite=True):
        path = _preset_path(str(preset_name))
        if path.exists() and not bool(overwrite):
            raise FileExistsError(f"SHARP camera preset exists: {path}")
        payload = {"type": "VHS_SHARP_CAMERA", "saved_at": time.time(), "camera": camera}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sharp_engine.log_info(f"Camera preset saved: {path}")
        return (str(path), json.dumps(payload, ensure_ascii=False, indent=2))


class VHSSharpLoadCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset_name_or_path": ("STRING", {"default": "my_sharp_camera"})}}

    RETURN_TYPES = (T_SHARP_CAMERA, "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "load"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def load(self, preset_name_or_path):
        raw = str(preset_name_or_path).strip()
        path = Path(raw).expanduser() if raw.endswith(".json") or os.path.sep in raw else _preset_path(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        camera = payload.get("camera", payload)
        if not isinstance(camera, dict):
            raise ValueError(f"Invalid SHARP camera preset: {path}")
        return (camera, _camera_info(camera))


class VHSSharpReferenceVideoToCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"reference_frames": ("IMAGE",)},
            "optional": {
                "reference_depth": ("IMAGE",),
                "samples": ("INT", {"default": 8, "min": 2, "max": 60, "step": 1}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "path_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "interpolation": (["smooth", "linear"], {"default": "smooth"}),
                "loop": ("BOOLEAN", {"default": False}),
                "depth_estimator": (["da2", "da1", "da3", "depthpro", "zoedepth"], {"default": "da2"}),
                "depth_model_size": (["small", "base", "large", "giant"], {"default": "small"}),
                "da3_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "allow_luminance_depth": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (T_SHARP_CAMERA, "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("camera", "path_json", "depth", "info")
    FUNCTION = "extract"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/SHARP VideoOps"

    def extract(self, reference_frames, reference_depth=None, **kwargs):
        sharp_engine.log_info(
            f"Node ReferenceVideoToCamera start: frames={int(reference_frames.shape[0])}, samples={kwargs.get('samples', 8)}"
        )
        keyframes = _estimate_reference_motion(
            reference_frames,
            int(kwargs.get("samples", 8)),
            float(kwargs.get("motion_strength", 1.0)),
        )
        if reference_depth is not None:
            depth = prepare_depth_map(reference_depth.detach().cpu().numpy()[0, :, :, 0], _image_tensor_to_np(reference_frames, 0))
        else:
            depth = estimate_depth(
                _image_tensor_to_np(reference_frames, 0),
                str(kwargs.get("depth_estimator", "da2")),
                model_size=str(kwargs.get("depth_model_size", "small")),
                da3_resolution=int(kwargs.get("da3_resolution", 1024)),
                allow_luminance_fallback=bool(kwargs.get("allow_luminance_depth", False)),
            )
        path_json = json.dumps(keyframes, ensure_ascii=False, indent=2)
        camera = {
            "preset": "keyframes",
            "keyframes": _parse_camera_path(path_json or DEFAULT_CAMERA_PATH),
            "path_scale": float(kwargs.get("path_scale", 1.0)),
            "interpolation": str(kwargs.get("interpolation", "smooth")),
            "loop": bool(kwargs.get("loop", False)),
            "lookat_mode": "point",
            "target_anchor": "center",
            "amplitude": 1.0,
            "radius_scale": 1.0,
        }
        info = {"method": "opencv_affine_motion_plus_depthflow_depth", "keyframes": len(keyframes)}
        return (camera, path_json, _depth_to_tensor(depth), json.dumps(info, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "VHS_SHARP_LoadPlyScene": VHSSharpLoadPlyScene,
    "VHS_SHARP_SceneProcess": VHSSharpSceneProcess,
    "VHS_SHARP_MergeScenes": VHSSharpMergeScenes,
    "VHS_SHARP_MergePlyFolder": VHSSharpMergePlyFolder,
    "VHS_SHARP_SaveCameraPreset": VHSSharpSaveCameraPreset,
    "VHS_SHARP_LoadCameraPreset": VHSSharpLoadCameraPreset,
    "VHS_SHARP_ReferenceVideoToCamera": VHSSharpReferenceVideoToCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_SHARP_LoadPlyScene": "SHARP Load PLY Scene 🍎📦🅥🅗🅢",
    "VHS_SHARP_SceneProcess": "SHARP Process 3DGS Scene 🍎🛠️🅥🅗🅢",
    "VHS_SHARP_MergeScenes": "SHARP Merge 3DGS Scenes 🍎🧩🅥🅗🅢",
    "VHS_SHARP_MergePlyFolder": "SHARP Merge PLY Folder 🍎📚🅥🅗🅢",
    "VHS_SHARP_SaveCameraPreset": "SHARP Save Camera Preset 🍎💾🅥🅗🅢",
    "VHS_SHARP_LoadCameraPreset": "SHARP Load Camera Preset 🍎📂🅥🅗🅢",
    "VHS_SHARP_ReferenceVideoToCamera": "SHARP Reference Video To Camera 🍎🎞️🅥🅗🅢",
}
