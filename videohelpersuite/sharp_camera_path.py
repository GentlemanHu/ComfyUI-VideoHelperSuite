"""Camera path helpers for SHARP VideoOps nodes."""
from __future__ import annotations

from typing import Any

import numpy as np


def sample_keyframe_eye(rig: dict[str, Any], t: float, offset_xyz: np.ndarray) -> list[float] | None:
    keyframes = rig.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        return None
    points: list[tuple[float, float, float, float]] = []
    for item in keyframes:
        try:
            if isinstance(item, dict):
                points.append((float(item["t"]), float(item["x"]), float(item["y"]), float(item["z"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                points.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
        except Exception:
            continue
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    t = float(t) % 1.0 if bool(rig.get("loop", False)) and len(points) > 1 else float(t)
    if t <= points[0][0]:
        selected = points[0][1:]
    elif t >= points[-1][0]:
        selected = points[-1][1:]
    else:
        selected = points[-1][1:]
        for left, right in zip(points, points[1:]):
            if left[0] <= t <= right[0]:
                span = max(right[0] - left[0], 1e-6)
                mix = (t - left[0]) / span
                if str(rig.get("interpolation", "smooth")) == "smooth":
                    mix = mix * mix * (3.0 - 2.0 * mix)
                selected = tuple(left[i] * (1.0 - mix) + right[i] * mix for i in range(1, 4))
                break
    scale = float(rig.get("path_scale", 1.0))
    return [
        float(selected[0]) * float(offset_xyz[0]) * scale,
        float(selected[1]) * float(offset_xyz[1]) * scale,
        float(selected[2]) * float(offset_xyz[2]) * scale,
    ]


def target_position(rig: dict[str, Any], offset_xyz: np.ndarray, focus: float) -> list[float]:
    anchors = {
        "center": (0.0, 0.0, 0.0),
        "left": (-0.35, 0.0, 0.0),
        "right": (0.35, 0.0, 0.0),
        "top": (0.0, 0.25, 0.0),
        "bottom": (0.0, -0.25, 0.0),
        "foreground": (0.0, 0.0, -0.35),
        "background": (0.0, 0.0, 0.35),
        "custom": (
            float(rig.get("target_x", 0.0)),
            float(rig.get("target_y", 0.0)),
            float(rig.get("target_z", 0.0)),
        ),
    }
    anchor = anchors.get(str(rig.get("target_anchor", "center")), anchors["center"])
    return [
        float(anchor[0]) * float(offset_xyz[0]),
        float(anchor[1]) * float(offset_xyz[1]),
        float(focus) + float(anchor[2]) * float(offset_xyz[2]),
    ]
