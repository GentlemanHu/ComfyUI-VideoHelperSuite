from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


if os.environ.get("VHS_DEPTHFLOW_ESTIMATE_DEVICE", "").lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def _setup_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    for base in (
        root / ".venv_depthflow",
        root.parent.parent / "DepthFlow",
        root.parent.parent / "ShaderFlow",
    ):
        if not base.exists():
            continue
        site_dirs = []
        if base.name.startswith(".venv"):
            site_dirs.extend(base.glob("lib/python*/site-packages"))
            site_dirs.extend(base.glob("Lib/site-packages"))
        for item in (base, *site_dirs):
            item_str = str(item)
            if item.exists() and item_str not in sys.path:
                sys.path.insert(0, item_str)


def _estimate(img_np: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    _setup_paths()
    key = str(params.get("estimator") or "da2").lower()
    postprocess = bool(params.get("postprocess", True))

    if key in ("da1", "da2", "da3"):
        from depthflow.estimators.anything import (
            DepthAnythingV1,
            DepthAnythingV2,
            DepthAnythingV3,
        )
        cls = {"da1": DepthAnythingV1, "da2": DepthAnythingV2, "da3": DepthAnythingV3}[key]
        kwargs: dict[str, Any] = {"post": postprocess}
        if hasattr(cls, "Model"):
            model_size = str(params.get("model_size") or "small")
            safe_size = "large" if key != "da3" and model_size == "giant" else model_size
            kwargs["model"] = cls.Model(safe_size)
        if key == "da3":
            kwargs["resolution"] = int(params.get("da3_resolution") or 1024)
        if params.get("sigma") is not None:
            kwargs["sigma"] = float(params["sigma"])
        if params.get("thicken") is not None:
            kwargs["thicken"] = int(params["thicken"])
        return np.asarray(cls(**kwargs).estimate(img_np), dtype=np.float32)

    if key == "depthpro":
        from depthflow.estimators.depthpro import DepthPro
        return np.asarray(DepthPro(post=postprocess).estimate(img_np), dtype=np.float32)

    if key == "zoedepth":
        from depthflow.estimators.zoedepth import ZoeDepth
        return np.asarray(ZoeDepth(post=postprocess).estimate(img_np), dtype=np.float32)

    if key == "marigold":
        from depthflow.estimators.marigold import Marigold
        return np.asarray(Marigold(post=postprocess).estimate(img_np), dtype=np.float32)

    raise ValueError(f"Unknown DepthFlow estimator: {key}")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: depthflow_estimate_worker.py INPUT_NPY OUTPUT_NPY PARAMS_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    params_path = Path(sys.argv[3])
    params = json.loads(params_path.read_text(encoding="utf-8"))
    img_np = np.load(input_path)
    depth = _estimate(img_np, params)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, depth.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
