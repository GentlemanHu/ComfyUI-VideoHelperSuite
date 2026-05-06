"""Reusable scene operations for SHARP VideoOps nodes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

import folder_paths

from . import sharp_engine


def load_ply_scene(ply_path: str, source_image=None, focal_px: float = 0.0) -> sharp_engine.SharpScene:
    from .apple_sharp import gaussians as gauss_mod

    path = Path(str(ply_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SHARP PLY not found: {path}")
    gaussians, meta = gauss_mod.load_ply(path)
    width, height = int(meta.image_size[0]), int(meta.image_size[1])
    if focal_px > 0:
        focal = float(focal_px)
    else:
        focal = float(meta.focal_length_px)
    if source_image is not None and isinstance(source_image, torch.Tensor):
        image = source_image[0] if source_image.dim() == 4 else source_image
        image = image[..., :3].detach().float().cpu().clamp(0, 1)
        height, width = int(image.shape[0]), int(image.shape[1])
    else:
        image = torch.zeros((height, width, 3), dtype=torch.float32)
    return build_scene_from_gaussians(gaussians, focal, (width, height), image, str(path))


def build_scene_from_gaussians(
    gaussians: Any,
    focal_px: float,
    source_size: tuple[int, int],
    source_image: torch.Tensor,
    ply_path: str = "",
) -> sharp_engine.SharpScene:
    pts = gaussians.mean_vectors.reshape(-1, 3).detach().float().cpu()
    bounds_min = torch.quantile(pts, 0.01, dim=0)
    bounds_max = torch.quantile(pts, 0.99, dim=0)
    center = (bounds_min + bounds_max) * 0.5
    radius = float(torch.linalg.norm(bounds_max - bounds_min).item() * 0.65)
    return sharp_engine.SharpScene(
        gaussians=gaussians,
        focal_px=float(focal_px),
        source_size=(int(source_size[0]), int(source_size[1])),
        source_image=source_image.detach().float().cpu().clamp(0, 1),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        center=center,
        radius=max(radius, 0.5),
        ply_path=str(ply_path),
    )


def filter_scene(scene: sharp_engine.SharpScene, max_gaussians: int, min_opacity: float) -> sharp_engine.SharpScene:
    gaussians = sharp_engine.filter_gaussians(scene.gaussians, int(max_gaussians), float(min_opacity))
    return build_scene_from_gaussians(gaussians, scene.focal_px, scene.source_size, scene.source_image, scene.ply_path)


def transform_scene(
    scene: sharp_engine.SharpScene,
    translate_xyz: tuple[float, float, float],
    scale: float,
    rotate_y_degrees: float,
) -> sharp_engine.SharpScene:
    g = scene.gaussians
    points = g.mean_vectors.detach().clone().float()
    yaw = torch.tensor(float(rotate_y_degrees) * 3.141592653589793 / 180.0, dtype=torch.float32)
    c, s = torch.cos(yaw), torch.sin(yaw)
    rot = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
    offset = torch.tensor(translate_xyz, dtype=torch.float32).view(1, 1, 3)
    points = torch.matmul(points, rot.T) * float(scale) + offset
    gaussians = type(g)(
        mean_vectors=points,
        singular_values=g.singular_values.detach().clone().float() * float(scale),
        quaternions=g.quaternions.detach().clone().float(),
        colors=g.colors.detach().clone().float(),
        opacities=g.opacities.detach().clone().float(),
    )
    return build_scene_from_gaussians(gaussians, scene.focal_px, scene.source_size, scene.source_image, scene.ply_path)


def merge_scenes(scenes: list[sharp_engine.SharpScene], max_gaussians: int = 0) -> sharp_engine.SharpScene:
    if not scenes:
        raise ValueError("SHARP merge needs at least one scene")
    first = scenes[0]
    mean_vectors = torch.cat([s.gaussians.mean_vectors.reshape(1, -1, 3) for s in scenes], dim=1)
    singular_values = torch.cat([s.gaussians.singular_values.reshape(1, -1, 3) for s in scenes], dim=1)
    quaternions = torch.cat([s.gaussians.quaternions.reshape(1, -1, 4) for s in scenes], dim=1)
    colors = torch.cat([s.gaussians.colors.reshape(1, -1, 3) for s in scenes], dim=1)
    opacities = torch.cat([s.gaussians.opacities.reshape(1, -1) for s in scenes], dim=1)
    gaussians = type(first.gaussians)(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )
    if int(max_gaussians) > 0:
        gaussians = sharp_engine.filter_gaussians(gaussians, int(max_gaussians), 0.0)
    return build_scene_from_gaussians(gaussians, first.focal_px, first.source_size, first.source_image, "")


def save_scene_ply(scene: sharp_engine.SharpScene, output_prefix: str) -> str:
    from .apple_sharp import gaussians as gauss_mod

    out_dir = Path(folder_paths.get_output_directory()) / "sharp_videoops"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_prefix}_{int(time.time() * 1000)}.ply"
    width, height = scene.source_size
    gauss_mod.save_ply(scene.gaussians, scene.focal_px, (height, width), out_path)
    sharp_engine.log_info(f"PLY saved: {out_path}")
    return str(out_path)


def scene_info(scene: sharp_engine.SharpScene) -> str:
    return json.dumps(
        {
            "gaussians": int(scene.gaussians.mean_vectors.reshape(-1, 3).shape[0]),
            "source_size": list(scene.source_size),
            "focal_px": float(scene.focal_px),
            "radius": float(scene.radius),
            "ply_path": scene.ply_path,
        },
        ensure_ascii=False,
        indent=2,
    )
