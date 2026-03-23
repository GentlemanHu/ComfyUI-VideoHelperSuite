import copy
import datetime
import wave
from pathlib import Path
from typing import Any

import movis as mv
import numpy as np
import torch
from PIL import Image
from PIL import ImageColor
from comfy.utils import ProgressBar
from tqdm import tqdm

import folder_paths

from .caption import resolve_media_path
from .utils import get_audio

try:
    from notifier.notify import notifyAll
except Exception:
    notifyAll = None


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


def _split_paths(value: str) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(";", ",")
    return [x.strip() for x in normalized.split(",") if x.strip()]

def _parse_float_list(value: str) -> list[float]:
    if not value or not str(value).strip():
        return []
    out = []
    for x in _split_paths(str(value)):
        try:
            fv = float(x)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def _parse_str_list(value: str) -> list[str]:
    if not value or not str(value).strip():
        return []
    return [x.strip() for x in _split_paths(str(value)) if x.strip()]


def _parse_bool_list(value: str) -> list[bool]:
    if not value or not str(value).strip():
        return []
    out: list[bool] = []
    for x in _split_paths(str(value)):
        v = str(x).strip().lower()
        out.append(v in {"1", "true", "t", "yes", "y", "on"})
    return out

def _float_or_default(value: str, default: float) -> float:
    s = str(value or "").strip()
    if not s:
        return float(default)
    return float(s)


def _safe_float(value: Any, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return out


def _resolve_motion_easings(easings: list[str], segment_count: int) -> list[str] | None:
    if segment_count <= 0:
        return None
    if not easings:
        return None
    out = [str(x) for x in easings[:segment_count]]
    while len(out) < segment_count:
        out.append(out[-1])
    return out


def _tensor_to_uint8(image_tensor):
    arr = image_tensor.cpu().numpy() * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_images(images, prefix: str) -> list[str]:
    temp_dir = Path(folder_paths.get_temp_directory()) / "vhs_movis_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, image in enumerate(images):
        p = temp_dir / f"{prefix}_{idx:05}.png"
        Image.fromarray(_tensor_to_uint8(image)).save(p, compress_level=3)
        out.append(str(p.resolve()))
    return out


def _save_single_image(image, prefix: str) -> str:
    temp_dir = Path(folder_paths.get_temp_directory()) / "vhs_movis_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor) and image.ndim >= 4:
        image = image[0]
    p = temp_dir / f"{prefix}_{_now_stamp()}.png"
    Image.fromarray(_tensor_to_uint8(image)).save(p, compress_level=3)
    return str(p.resolve())


def _resolve_image_input(image_path: str, image=None, prefix: str = "movis_image") -> str:
    if isinstance(image_path, str) and image_path.strip():
        path = resolve_media_path(image_path, must_exist=True)
        if not _is_image(path):
            raise ValueError("image_path 不是图片文件")
        return path
    if image is not None:
        return _save_single_image(image, prefix)
    raise ValueError("未提供有效图片输入：请设置 image_path 或连接 IMAGE")


def _parse_rgba(color: str, default=(0, 0, 0, 0)):
    try:
        return ImageColor.getcolor(str(color), "RGBA")
    except Exception:
        return default


def _normalize_anchor(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return float(default)

def _position_value_to_pixels(value: Any, canvas_size: int) -> float:
    value = _safe_float(value, 0.5)
    if 0.0 <= value <= 1.0:
        return float(value * canvas_size)
    return float(value)

def _position_to_pixels(x: Any, y: Any, width: int, height: int) -> tuple[float, float]:
    return (
        _position_value_to_pixels(x, int(width)),
        _position_value_to_pixels(y, int(height)),
    )


def _prepare_image_for_canvas(path: str, width: int, height: int, mode: str, anchor_x: float, anchor_y: float, bg_color: str) -> str:
    mode = str(mode or "transform_only")
    if mode == "transform_only":
        return path

    src = Image.open(path).convert("RGBA")
    sw, sh = src.size
    cw, ch = int(width), int(height)
    if sw <= 0 or sh <= 0 or cw <= 0 or ch <= 0:
        return path

    ax = _normalize_anchor(anchor_x, 0.5)
    ay = _normalize_anchor(anchor_y, 0.5)
    bg = _parse_rgba(bg_color, (0, 0, 0, 0))

    if mode == "stretch":
        out = src.resize((cw, ch), Image.Resampling.LANCZOS)
    elif mode in {"cover", "center_crop"}:
        scale = max(cw / sw, ch / sh)
        tw, th = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        rs = src.resize((tw, th), Image.Resampling.LANCZOS)
        left = int(round((tw - cw) * ax))
        top = int(round((th - ch) * ay))
        left = max(0, min(left, max(0, tw - cw)))
        top = max(0, min(top, max(0, th - ch)))
        out = rs.crop((left, top, left + cw, top + ch))
    else:
        # contain / center_inside
        if mode == "center_inside":
            scale = min(1.0, min(cw / sw, ch / sh))
        else:
            scale = min(cw / sw, ch / sh)
        tw, th = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        rs = src.resize((tw, th), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (cw, ch), bg)
        x = int(round((cw - tw) * ax))
        y = int(round((ch - th) * ay))
        x = max(0, min(x, max(0, cw - tw)))
        y = max(0, min(y, max(0, ch - th)))
        out.alpha_composite(rs, (x, y))

    temp_dir = Path(folder_paths.get_temp_directory()) / "vhs_movis_images" / "prepared"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"prepared_{_now_stamp()}.png"
    out.save(out_path, compress_level=3)
    return str(out_path.resolve())


def _audio_to_wav_file(audio: dict[str, Any], prefix: str = "movis_audio") -> str:
    """将 ComfyUI AUDIO 转为临时 wav 文件，供 movis 读取。"""
    if not audio or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO 输入格式无效，缺少 waveform/sample_rate")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("AUDIO.sample_rate 必须大于 0")

    if isinstance(waveform, torch.Tensor):
        data = waveform.detach().cpu().numpy()
    else:
        data = np.asarray(waveform)

    # 兼容 [B, C, T] / [C, T] / [T]
    if data.ndim == 3:
        data = data[0]
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("AUDIO.waveform 维度不支持，期望 [B,C,T]/[C,T]/[T]")

    channels, samples = data.shape
    if samples <= 0:
        raise ValueError("AUDIO.waveform 为空")

    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    interleaved = pcm.T.reshape(-1)

    temp_dir = Path(folder_paths.get_temp_directory()) / "vhs_movis_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{prefix}_{_now_stamp()}.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())
    return str(out_path.resolve())


def _extract_video_paths_from_vhs_filenames(vhs_filenames: Any) -> list[str]:
    """从 VHS_FILENAMES 中提取可用视频路径。"""
    if vhs_filenames is None:
        return []

    items = []
    if isinstance(vhs_filenames, (tuple, list)) and len(vhs_filenames) >= 2 and isinstance(vhs_filenames[1], list):
        items = vhs_filenames[1]
    elif isinstance(vhs_filenames, list):
        items = vhs_filenames

    out = []
    for p in items:
        if isinstance(p, str) and p.strip():
            try:
                out.append(resolve_media_path(p, must_exist=True))
            except Exception:
                continue
    return out


def _resolve_video_input(video_path: str, vhs_filenames: Any = None) -> str:
    """优先使用 video_path；为空时回退到 VHS_FILENAMES 最后一个文件。"""
    if isinstance(video_path, str) and video_path.strip():
        return resolve_media_path(video_path, must_exist=True)
    candidates = _extract_video_paths_from_vhs_filenames(vhs_filenames)
    if candidates:
        return candidates[-1]
    raise ValueError("未提供有效视频输入：请设置 video_path 或连接 VHS_FILENAMES")


def _resolve_audio_input(audio_path: str, audio: Any = None, prefix: str = "movis_audio") -> str:
    """优先使用 audio_path；为空时回退到 ComfyUI AUDIO。"""
    if isinstance(audio_path, str) and audio_path.strip():
        return resolve_media_path(audio_path, must_exist=True)
    if audio is not None:
        return _audio_to_wav_file(audio, prefix=prefix)
    raise ValueError("未提供有效音频输入：请设置 audio_path 或连接 AUDIO")


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _video_duration(path: str) -> float:
    return float(mv.layer.Video(path, audio=False).duration)


def _audio_duration(path: str) -> float:
    return float(mv.layer.media.Audio(path).duration)


def _new_timeline(width: int, height: int, fps: float, bg_color: str) -> dict[str, Any]:
    return {
        "canvas": {
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "bg_color": bg_color,
        },
        "video_tracks": [],
        "audio_tracks": [],
        "text_tracks": [],
        "bgm": None,
    }


def _clone_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(timeline)

def _offset_timeline(timeline: dict[str, Any], offset: float) -> dict[str, Any]:
    t = _clone_timeline(timeline)
    o = float(offset)
    if o == 0.0:
        return t
    for clip in t.get("video_tracks", []):
        clip["start"] = float(clip.get("start", 0.0)) + o
    for track in t.get("audio_tracks", []):
        track["start"] = float(track.get("start", 0.0)) + o
    for txt in t.get("text_tracks", []):
        txt["start"] = float(txt.get("start", 0.0)) + o
    return t


def _timeline_duration(timeline: dict[str, Any]) -> float:
    end_points = [1.0]
    for clip in timeline.get("video_tracks", []):
        end_points.append(float(clip["start"]) + float(clip["duration"]))
    for track in timeline.get("audio_tracks", []):
        end_points.append(float(track["start"]) + float(track["duration"]))
    for txt in timeline.get("text_tracks", []):
        end_points.append(float(txt["start"]) + float(txt["duration"]))
    bgm = timeline.get("bgm")
    if bgm and bgm.get("path"):
        end_points.append(float(bgm.get("duration") or 0))
    return max(end_points)

def _timeline_content_duration(timeline: dict[str, Any]) -> float:
    end_points = [0.0]
    for clip in timeline.get("video_tracks", []):
        end_points.append(float(clip.get("start", 0.0)) + float(clip.get("duration", 0.0)))
    for track in timeline.get("audio_tracks", []):
        end_points.append(float(track.get("start", 0.0)) + float(track.get("duration", 0.0)))
    for txt in timeline.get("text_tracks", []):
        end_points.append(float(txt.get("start", 0.0)) + float(txt.get("duration", 0.0)))
    return max(end_points)


def _smart_bgm_level(timeline: dict[str, Any], default_bgm_db: float = -12.0, duck_db: float = -18.0) -> float:
    """若存在前景音轨，则自动压低 BGM。"""
    bgm = timeline.get("bgm") or {}
    base = float(bgm.get("audio_level_db", default_bgm_db))
    has_foreground_audio = len(timeline.get("audio_tracks", [])) > 0 or any(
        bool(c.get("use_source_audio", False)) for c in timeline.get("video_tracks", [])
    )
    if has_foreground_audio:
        return min(base, float(duck_db))
    return base


def _render_timeline(timeline: dict[str, Any], output_file_prefix: str, notify_all: bool):
    canvas = timeline["canvas"]
    width = int(_safe_float(canvas.get("width", 1080), 1080.0, min_value=64.0, max_value=8192.0))
    height = int(_safe_float(canvas.get("height", 1920), 1920.0, min_value=64.0, max_value=8192.0))
    fps = _safe_float(canvas.get("fps", 30.0), 30.0, min_value=1.0, max_value=240.0)
    bg_color = str(canvas.get("bg_color", "#000000"))
    duration = _safe_float(_timeline_duration(timeline), 1.0, min_value=0.01)

    scene = mv.layer.Composition(size=(width, height), duration=duration)
    render_frame_count = max(1, len(np.arange(0.0, duration, 1.0 / fps)))
    total_steps = 1 + len(timeline.get("video_tracks", [])) + len(timeline.get("audio_tracks", [])) + len(timeline.get("text_tracks", [])) + (1 if timeline.get("bgm") else 0) + render_frame_count + 1
    pbar = ProgressBar(max(1, total_steps))
    scene.add_layer(
        mv.layer.Rectangle(size=(width, height), color=bg_color, duration=duration),
        name="background",
    )
    pbar.update(1)

    for clip in timeline.get("video_tracks", []):
        start = _safe_float(clip.get("start", 0.0), 0.0, min_value=0.0)
        clip_duration = _safe_float(clip.get("duration", 0.01), 0.01, min_value=0.01)
        source_start = _safe_float(clip.get("source_start", 0.0), 0.0, min_value=0.0)
        fade_in = _safe_float(clip.get("fade_in", 0.0), 0.0, min_value=0.0)
        fade_out = _safe_float(clip.get("fade_out", 0.0), 0.0, min_value=0.0)
        position = _position_to_pixels(
            clip.get("position_x", 0.5),
            clip.get("position_y", 0.5),
            width,
            height,
        )
        scale = (
            _safe_float(clip.get("scale_x", 1.0), 1.0, min_value=0.01, max_value=20.0),
            _safe_float(clip.get("scale_y", 1.0), 1.0, min_value=0.01, max_value=20.0),
        )
        rotation = _safe_float(clip.get("rotation", 0.0), 0.0, min_value=-360.0, max_value=360.0)
        opacity = _safe_float(clip.get("opacity", 1.0), 1.0, min_value=0.0, max_value=1.0)
        motion = clip.get("motion") if isinstance(clip.get("motion"), dict) else {}
        motion_applied_opacity = False

        if clip.get("is_image", False):
            prepared_path = _prepare_image_for_canvas(
                clip["path"],
                width,
                height,
                clip.get("image_scale_mode", "transform_only"),
                clip.get("image_anchor_x", 0.5),
                clip.get("image_anchor_y", 0.5),
                clip.get("image_bg_color", "#00000000"),
            )
            layer = mv.layer.Image(prepared_path, duration=clip_duration)
            item = scene.add_layer(
                layer,
                offset=start,
                position=position,
                scale=scale,
                rotation=rotation,
                opacity=opacity,
            )
        else:
            layer = mv.layer.Video(clip["path"], audio=bool(clip.get("use_source_audio", True)))
            item = scene.add_layer(
                layer,
                offset=start,
                start_time=source_start,
                end_time=source_start + clip_duration,
                position=position,
                scale=scale,
                rotation=rotation,
                opacity=opacity,
                audio=bool(clip.get("use_source_audio", True)),
                audio_level=float(clip.get("audio_level_db", 0.0)),
            )

        keyframe_times = motion.get("keyframe_times") if isinstance(motion, dict) else None
        if isinstance(keyframe_times, list) and len(keyframe_times) >= 2:
            norm_times = [max(0.0, min(1.0, float(x))) for x in keyframe_times]
            motion_times = [float(clip_duration) * t for t in norm_times]
            segment_count = max(0, len(motion_times) - 1)
            easings = _resolve_motion_easings(
                [str(x) for x in motion.get("easings", [])] if isinstance(motion.get("easings"), list) else [],
                segment_count,
            )

            px = motion.get("position_x", []) if isinstance(motion.get("position_x"), list) else []
            py = motion.get("position_y", []) if isinstance(motion.get("position_y"), list) else []
            sx = motion.get("scale_x", []) if isinstance(motion.get("scale_x"), list) else []
            sy = motion.get("scale_y", []) if isinstance(motion.get("scale_y"), list) else []
            rt = motion.get("rotation", []) if isinstance(motion.get("rotation"), list) else []
            op = motion.get("opacity", []) if isinstance(motion.get("opacity"), list) else []

            if px or py:
                pos_values = []
                for i in range(len(motion_times)):
                    raw_x = _safe_float(px[i], 0.5) if i < len(px) else _safe_float(clip.get("position_x", 0.5), 0.5)
                    raw_y = _safe_float(py[i], 0.5) if i < len(py) else _safe_float(clip.get("position_y", 0.5), 0.5)
                    pos_values.append(_position_to_pixels(raw_x, raw_y, width, height))
                item.position.enable_motion().clear().extend(
                    keyframes=motion_times,
                    values=pos_values,
                    easings=easings,
                )

            if sx or sy:
                scale_values = []
                for i in range(len(motion_times)):
                    x = _safe_float(sx[i], scale[0], min_value=0.01, max_value=20.0) if i < len(sx) else scale[0]
                    y = _safe_float(sy[i], scale[1], min_value=0.01, max_value=20.0) if i < len(sy) else scale[1]
                    scale_values.append((x, y))
                item.scale.enable_motion().clear().extend(
                    keyframes=motion_times,
                    values=scale_values,
                    easings=easings,
                )

            if rt:
                rot_values = []
                for i in range(len(motion_times)):
                    rot_values.append(_safe_float(rt[i], rotation, min_value=-360.0, max_value=360.0) if i < len(rt) else rotation)
                item.rotation.enable_motion().clear().extend(
                    keyframes=motion_times,
                    values=rot_values,
                    easings=easings,
                )

            if op:
                opacity_values = []
                for i in range(len(motion_times)):
                    opacity_values.append(_safe_float(op[i], opacity, min_value=0.0, max_value=1.0) if i < len(op) else opacity)
                item.opacity.enable_motion().clear().extend(
                    keyframes=motion_times,
                    values=opacity_values,
                    easings=easings,
                )
                motion_applied_opacity = True

        if fade_in > 0 and not motion_applied_opacity:
            item.opacity.enable_motion().extend(keyframes=[0.0, fade_in], values=[0.0, opacity])
        if fade_out > 0 and not motion_applied_opacity:
            t0 = max(0.0, clip_duration - fade_out)
            item.opacity.enable_motion().extend(keyframes=[t0, clip_duration], values=[opacity, 0.0])
        pbar.update(1)

    for track in timeline.get("audio_tracks", []):
        source_start = float(track.get("source_start", 0.0))
        audio_clip = mv.layer.media.Audio(track["path"])
        scene.add_layer(
            audio_clip,
            offset=float(track["start"]),
            start_time=source_start,
            end_time=source_start + float(track["duration"]),
            audio_level=float(track.get("audio_level_db", 0.0)),
        )
        pbar.update(1)

    for txt in timeline.get("text_tracks", []):
        layer = mv.layer.Text(
            text=str(txt["text"]),
            font_size=float(txt.get("font_size", 64)),
            font_family=str(txt.get("font_family", "Sans Serif")),
            color=str(txt.get("color", "white")),
            duration=float(txt["duration"]),
        )
        scene.add_layer(
            layer,
            offset=float(txt["start"]),
            position=_position_to_pixels(
                txt.get("position_x", 0.5),
                txt.get("position_y", 0.9),
                width,
                height,
            ),
            opacity=float(txt.get("opacity", 1.0)),
        )
        pbar.update(1)

    bgm = timeline.get("bgm")
    if bgm and bgm.get("path"):
        source_start = float(bgm.get("source_start", 0.0))
        scene.add_layer(
            mv.layer.media.Audio(bgm["path"]),
            offset=0.0,
            start_time=source_start,
            end_time=source_start + duration,
            audio_level=float(_smart_bgm_level(timeline, default_bgm_db=float(bgm.get("audio_level_db", -12.0)))),
        )
        pbar.update(1)

    out_name = f"{output_file_prefix}{_now_stamp()}.mp4"
    out_path = str((Path(folder_paths.get_output_directory()) / out_name).resolve())

    print(f"[VHS_MOVIS] render start -> {out_path} | {width}x{height}@{fps:.2f} | duration={duration:.3f}s")

    original_write_video = mv.layer.Composition._write_video

    class _WriterProxy:
        def __init__(self, real_writer):
            self._real_writer = real_writer

        def append_data(self, frame):
            self._real_writer.append_data(frame)
            pbar.update(1)

        def close(self):
            return self._real_writer.close()

        def __getattr__(self, item):
            return getattr(self._real_writer, item)

    def _write_video_with_progress(self, start_time: float, end_time: float, fps_value: float, writer):
        # 保留 movis 原生 tqdm 控制台日志，同时通过 writer 代理更新 ComfyUI 进度条
        return original_write_video(self, start_time, end_time, fps_value, _WriterProxy(writer))

    mv.layer.Composition._write_video = _write_video_with_progress
    try:
        scene.write_video(
            out_path,
            codec="libx264",
            pixelformat="yuv420p",
            fps=fps,
            audio=True,
            output_params=["-movflags", "+faststart"],
        )
    finally:
        mv.layer.Composition._write_video = original_write_video

    pbar.update(1)
    print(f"[VHS_MOVIS] render done -> {out_path}")
    frames = max(1, int(round(duration * fps)))
    try:
        audio = get_audio(out_path)
    except Exception:
        # 某些组合（例如全静音时间线）可能不会生成音轨。
        # 为保持节点返回类型稳定，回退为空 AUDIO，避免整条工作流失败。
        audio = {
            "waveform": torch.zeros((1, 2, 0), dtype=torch.float32),
            "sample_rate": 44100,
        }
    if notify_all and notifyAll:
        notifyAll(out_path, "movis_pipeline")
    return out_path, audio, frames, duration


class MovisCreateTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192, "tooltip": "输出画布宽度"}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192, "tooltip": "输出画布高度"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "tooltip": "输出帧率"}),
                "bg_color": ("STRING", {"default": "#000000", "tooltip": "背景色，如 #000000"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "create"

    def create(self, width, height, fps, bg_color):
        safe_w = int(_safe_float(width, 1080.0, min_value=64.0, max_value=8192.0))
        safe_h = int(_safe_float(height, 1920.0, min_value=64.0, max_value=8192.0))
        safe_fps = _safe_float(fps, 30.0, min_value=1.0, max_value=240.0)
        return (_new_timeline(safe_w, safe_h, safe_fps, str(bg_color or "#000000")),)


class MovisAddVideoTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "video_path": ("STRING", {"default": "", "placeholder": "可留空，改接 VHS_FILENAMES", "tooltip": "视频路径；为空时回退到VHS_FILENAMES"}),
                "placement_mode": (["append", "absolute"], {"default": "append", "tooltip": "append 自动接到当前时间线末尾；absolute 严格使用 start"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "append 模式为相对末尾偏移；absolute 模式为绝对开始秒数"}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "片段时长；0表示自动使用源时长"}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "源视频起始秒"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡入时长"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡出时长"}),
                "use_source_audio": ("BOOLEAN", {"default": True, "tooltip": "是否启用源视频音频"}),
                "audio_level_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0, "tooltip": "源音频增益(dB)"}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "画布Y位置（0~1 归一化）"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "tooltip": "水平缩放倍数"}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "tooltip": "垂直缩放倍数"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "tooltip": "旋转角度（度）"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "透明度（0~1）"}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_video"

    def add_video(self, timeline, video_path, placement_mode, start, duration, source_start, fade_in, fade_out, use_source_audio, audio_level_db, position_x, position_y, scale_x, scale_y, rotation, opacity, vhs_filenames=None):
        t = _clone_timeline(timeline)
        path = _resolve_video_input(video_path, vhs_filenames=vhs_filenames)
        clip_duration = _safe_float(duration, 0.0, min_value=0.0)
        if clip_duration <= 0:
            clip_duration = _video_duration(path)
        start_base = _timeline_content_duration(t) if str(placement_mode) == "append" else 0.0
        resolved_start = start_base + _safe_float(start, 0.0, min_value=0.0)
        t["video_tracks"].append(
            {
                "path": path,
                "is_image": False,
                "start": resolved_start,
                "duration": max(0.01, clip_duration),
                "source_start": _safe_float(source_start, 0.0, min_value=0.0),
                "fade_in": _safe_float(fade_in, 0.0, min_value=0.0),
                "fade_out": _safe_float(fade_out, 0.0, min_value=0.0),
                "use_source_audio": bool(use_source_audio),
                "audio_level_db": _safe_float(audio_level_db, 0.0, min_value=-60.0, max_value=24.0),
                "position_x": _safe_float(position_x, 0.5),
                "position_y": _safe_float(position_y, 0.5),
                "scale_x": _safe_float(scale_x, 1.0, min_value=0.01, max_value=20.0),
                "scale_y": _safe_float(scale_y, 1.0, min_value=0.01, max_value=20.0),
                "rotation": _safe_float(rotation, 0.0, min_value=-360.0, max_value=360.0),
                "opacity": _safe_float(opacity, 1.0, min_value=0.0, max_value=1.0),
            }
        )
        return (t,)


class MovisAddImageTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "image_path": ("STRING", {"default": "", "placeholder": "图片路径，可留空改接 IMAGE", "tooltip": "支持路径或原生IMAGE输入；优先使用image_path"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "duration": ("FLOAT", {"default": 2.0, "min": 0.01}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "画布X位置（0~1 归一化）"}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "画布Y位置（0~1 归一化）"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "tooltip": "水平缩放倍数"}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "tooltip": "垂直缩放倍数"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "tooltip": "旋转角度（度）"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "透明度（0~1）"}),
                "image_scale_mode": (["transform_only", "contain", "cover", "center_inside", "stretch"], {"default": "transform_only", "tooltip": "图片适配画布策略"}),
                "image_anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "水平锚点；cover/contain时决定裁剪或摆放焦点"}),
                "image_anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "垂直锚点；cover/contain时决定裁剪或摆放焦点"}),
                "image_bg_color": ("STRING", {"default": "#00000000", "tooltip": "contain/center_inside时的填充背景色"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_image"

    def add_image(self, timeline, image_path, start, duration, fade_in, fade_out, position_x, position_y, scale_x, scale_y, rotation, opacity, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color, image=None):
        t = _clone_timeline(timeline)
        path = _resolve_image_input(image_path, image=image, prefix="movis_add_image")
        t["video_tracks"].append(
            {
                "path": path,
                "is_image": True,
                "start": _safe_float(start, 0.0, min_value=0.0),
                "duration": _safe_float(duration, 2.0, min_value=0.01),
                "source_start": 0.0,
                "fade_in": _safe_float(fade_in, 0.0, min_value=0.0),
                "fade_out": _safe_float(fade_out, 0.0, min_value=0.0),
                "use_source_audio": False,
                "audio_level_db": 0.0,
                "position_x": _safe_float(position_x, 0.5),
                "position_y": _safe_float(position_y, 0.5),
                "scale_x": _safe_float(scale_x, 1.0, min_value=0.01, max_value=20.0),
                "scale_y": _safe_float(scale_y, 1.0, min_value=0.01, max_value=20.0),
                "rotation": _safe_float(rotation, 0.0, min_value=-360.0, max_value=360.0),
                "opacity": _safe_float(opacity, 1.0, min_value=0.0, max_value=1.0),
                "image_scale_mode": str(image_scale_mode),
                "image_anchor_x": _safe_float(image_anchor_x, 0.5, min_value=0.0, max_value=1.0),
                "image_anchor_y": _safe_float(image_anchor_y, 0.5, min_value=0.0, max_value=1.0),
                "image_bg_color": str(image_bg_color),
            }
        )
        return (t,)


class MovisMergeTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_a": ("MOVIS_TIMELINE",),
                "timeline_b": ("MOVIS_TIMELINE",),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -86400.0, "max": 86400.0}),
                "merge_bgm": (["keep_a", "use_b", "mix_as_audio_track"], {"default": "keep_a"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "merge"

    def merge(self, timeline_a, timeline_b, offset_b, merge_bgm):
        ta = _clone_timeline(timeline_a)
        tb = _offset_timeline(timeline_b, offset_b)

        ta["video_tracks"].extend(tb.get("video_tracks", []))
        ta["audio_tracks"].extend(tb.get("audio_tracks", []))
        ta["text_tracks"].extend(tb.get("text_tracks", []))

        bgm_b = tb.get("bgm")
        if merge_bgm == "use_b":
            ta["bgm"] = bgm_b
        elif merge_bgm == "mix_as_audio_track" and bgm_b and bgm_b.get("path"):
            ta["audio_tracks"].append(
                {
                    "path": bgm_b["path"],
                    "start": float(offset_b),
                    "duration": float(bgm_b.get("duration", 0.0) or 0.0),
                    "source_start": float(bgm_b.get("source_start", 0.0)),
                    "audio_level_db": float(bgm_b.get("audio_level_db", -12.0)),
                }
            )
        return (ta,)

class MovisAddImageSequenceTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "images": ("IMAGE",),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "序列起始时间（秒）"}),
                "seconds_per_image": ("FLOAT", {"default": 2.0, "min": 0.01, "tooltip": "默认每张图时长（秒）"}),
                "durations_csv": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：每张图时长列表，如 1.0,2.5,0.8"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡入时长（秒）"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡出时长（秒）"}),
                "image_scale_mode": (["transform_only", "contain", "cover", "center_inside", "stretch"], {"default": "transform_only", "tooltip": "图片适配画布策略"}),
                "image_anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "水平锚点"}),
                "image_anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "垂直锚点"}),
                "image_bg_color": ("STRING", {"default": "#00000000", "tooltip": "背景填充色（contain/center_inside）"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE", "INT", "FLOAT")
    RETURN_NAMES = ("timeline", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_images"

    def add_images(self, timeline, images, start, seconds_per_image, durations_csv, fade_in, fade_out, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color):
        t = _clone_timeline(timeline)
        image_files = _save_images(images, f"movis_img_seq_{_now_stamp()}")
        durations = _parse_float_list(durations_csv)
        cursor = float(start)
        total_duration = 0.0
        for idx, p in enumerate(image_files):
            dur = float(seconds_per_image)
            if idx < len(durations):
                dur = max(0.01, float(durations[idx]))
            t["video_tracks"].append(
                {
                    "path": p,
                    "is_image": True,
                    "start": cursor,
                    "duration": dur,
                    "source_start": 0.0,
                    "fade_in": float(fade_in),
                    "fade_out": float(fade_out),
                    "use_source_audio": False,
                    "audio_level_db": 0.0,
                    "position_x": 0.5,
                    "position_y": 0.5,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "rotation": 0.0,
                    "opacity": 1.0,
                    "image_scale_mode": str(image_scale_mode),
                    "image_anchor_x": float(image_anchor_x),
                    "image_anchor_y": float(image_anchor_y),
                    "image_bg_color": str(image_bg_color),
                }
            )
            cursor += dur
            total_duration += dur
        fps = float(t["canvas"]["fps"])
        return (t, max(1, int(round(total_duration * fps))), total_duration)


class MovisAddImageMotionTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "image_path": ("STRING", {"default": "", "placeholder": "单图路径，可留空改接 IMAGE", "tooltip": "支持路径或原生IMAGE输入；用于关键帧运动"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "片段起始时间（秒）"}),
                "duration": ("FLOAT", {"default": 3.0, "min": 0.01, "tooltip": "片段时长（秒）"}),
                "keyframe_times_csv": ("STRING", {"default": "0,1", "placeholder": "0~1 归一化时间点"}),
                "position_x_csv": ("STRING", {"default": "0.5,0.5", "tooltip": "X关键帧序列（逗号分隔，0~1优先按归一化）"}),
                "position_y_csv": ("STRING", {"default": "0.5,0.5", "tooltip": "Y关键帧序列（逗号分隔，0~1优先按归一化）"}),
                "scale_x_csv": ("STRING", {"default": "1.0,1.0", "tooltip": "scale_x关键帧序列"}),
                "scale_y_csv": ("STRING", {"default": "1.0,1.0", "tooltip": "scale_y关键帧序列"}),
                "rotation_csv": ("STRING", {"default": "0,0", "tooltip": "rotation关键帧序列（度）"}),
                "opacity_csv": ("STRING", {"default": "1,1", "tooltip": "opacity关键帧序列（0~1）"}),
                "easing_csv": ("STRING", {"default": "linear", "placeholder": "可选：linear,ease_in,ease_out"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡入时长（秒）"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡出时长（秒）"}),
                "image_scale_mode": (["transform_only", "contain", "cover", "center_inside", "stretch"], {"default": "transform_only", "tooltip": "图片适配画布策略"}),
                "image_anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "水平锚点"}),
                "image_anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "垂直锚点"}),
                "image_bg_color": ("STRING", {"default": "#00000000", "tooltip": "背景填充色（contain/center_inside）"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_motion"

    def add_motion(self, timeline, image_path, start, duration, keyframe_times_csv, position_x_csv, position_y_csv, scale_x_csv, scale_y_csv, rotation_csv, opacity_csv, easing_csv, fade_in, fade_out, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color, image=None):
        t = _clone_timeline(timeline)
        path = _resolve_image_input(image_path, image=image, prefix="movis_image_motion")

        kts = _parse_float_list(keyframe_times_csv)
        if len(kts) < 2:
            kts = [0.0, 1.0]
        kts = [max(0.0, min(1.0, x)) for x in kts]
        kts = sorted(kts)

        px = _parse_float_list(position_x_csv)
        py = _parse_float_list(position_y_csv)
        sx = _parse_float_list(scale_x_csv)
        sy = _parse_float_list(scale_y_csv)
        rt = _parse_float_list(rotation_csv)
        op = _parse_float_list(opacity_csv)
        easings = _parse_str_list(easing_csv)

        t["video_tracks"].append(
            {
                "path": path,
                "is_image": True,
                "start": float(start),
                "duration": max(0.01, float(duration)),
                "source_start": 0.0,
                "fade_in": float(fade_in),
                "fade_out": float(fade_out),
                "use_source_audio": False,
                "audio_level_db": 0.0,
                "position_x": px[0] if len(px) > 0 else _float_or_default("", 0.5),
                "position_y": py[0] if len(py) > 0 else _float_or_default("", 0.5),
                "scale_x": sx[0] if len(sx) > 0 else _float_or_default("", 1.0),
                "scale_y": sy[0] if len(sy) > 0 else _float_or_default("", 1.0),
                "rotation": rt[0] if len(rt) > 0 else _float_or_default("", 0.0),
                "opacity": op[0] if len(op) > 0 else _float_or_default("", 1.0),
                "image_scale_mode": str(image_scale_mode),
                "image_anchor_x": float(image_anchor_x),
                "image_anchor_y": float(image_anchor_y),
                "image_bg_color": str(image_bg_color),
                "motion": {
                    "keyframe_times": kts,
                    "position_x": px,
                    "position_y": py,
                    "scale_x": sx,
                    "scale_y": sy,
                    "rotation": rt,
                    "opacity": op,
                    "easings": easings,
                },
            }
        )
        return (t,)


class MovisAddVideoMotionTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "video_path": ("STRING", {"default": "", "placeholder": "可留空，改接 VHS_FILENAMES"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "片段起始时间（秒）"}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "片段时长；0表示自动读取源时长"}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "源视频截取起点（秒）"}),
                "keyframe_times_csv": ("STRING", {"default": "0,1", "placeholder": "0~1 归一化时间点"}),
                "position_x_csv": ("STRING", {"default": "0.5,0.5", "tooltip": "X关键帧序列（逗号分隔，0~1优先按归一化）"}),
                "position_y_csv": ("STRING", {"default": "0.5,0.5", "tooltip": "Y关键帧序列（逗号分隔，0~1优先按归一化）"}),
                "scale_x_csv": ("STRING", {"default": "1.0,1.0"}),
                "scale_y_csv": ("STRING", {"default": "1.0,1.0"}),
                "rotation_csv": ("STRING", {"default": "0,0"}),
                "opacity_csv": ("STRING", {"default": "1,1"}),
                "easing_csv": ("STRING", {"default": "linear", "placeholder": "可选：linear,ease_in,ease_out"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡入时长（秒）"}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "淡出时长（秒）"}),
                "use_source_audio": ("BOOLEAN", {"default": True, "tooltip": "是否使用源视频音轨"}),
                "audio_level_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0, "tooltip": "音轨增益（dB）"}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_video_motion"

    def add_video_motion(self, timeline, video_path, start, duration, source_start, keyframe_times_csv, position_x_csv, position_y_csv, scale_x_csv, scale_y_csv, rotation_csv, opacity_csv, easing_csv, fade_in, fade_out, use_source_audio, audio_level_db, vhs_filenames=None):
        t = _clone_timeline(timeline)
        path = _resolve_video_input(video_path, vhs_filenames=vhs_filenames)
        clip_duration = float(duration) if float(duration) > 0 else _video_duration(path)

        kts = _parse_float_list(keyframe_times_csv)
        if len(kts) < 2:
            kts = [0.0, 1.0]
        kts = sorted([max(0.0, min(1.0, x)) for x in kts])

        px = _parse_float_list(position_x_csv)
        py = _parse_float_list(position_y_csv)
        sx = _parse_float_list(scale_x_csv)
        sy = _parse_float_list(scale_y_csv)
        rt = _parse_float_list(rotation_csv)
        op = _parse_float_list(opacity_csv)
        easings = _parse_str_list(easing_csv)

        t["video_tracks"].append(
            {
                "path": path,
                "is_image": False,
                "start": float(start),
                "duration": max(0.01, clip_duration),
                "source_start": float(source_start),
                "fade_in": float(fade_in),
                "fade_out": float(fade_out),
                "use_source_audio": bool(use_source_audio),
                "audio_level_db": float(audio_level_db),
                "position_x": px[0] if len(px) > 0 else _float_or_default("", 0.5),
                "position_y": py[0] if len(py) > 0 else _float_or_default("", 0.5),
                "scale_x": sx[0] if len(sx) > 0 else _float_or_default("", 1.0),
                "scale_y": sy[0] if len(sy) > 0 else _float_or_default("", 1.0),
                "rotation": rt[0] if len(rt) > 0 else _float_or_default("", 0.0),
                "opacity": op[0] if len(op) > 0 else _float_or_default("", 1.0),
                "motion": {
                    "keyframe_times": kts,
                    "position_x": px,
                    "position_y": py,
                    "scale_x": sx,
                    "scale_y": sy,
                    "rotation": rt,
                    "opacity": op,
                    "easings": easings,
                },
            }
        )
        return (t,)

class MovisAddAudioTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "audio_path": ("STRING", {"default": "", "placeholder": "可留空，改接 AUDIO"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "audio_level_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_audio"

    def add_audio(self, timeline, audio_path, start, duration, source_start, audio_level_db, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_audio_input(audio_path, audio=audio, prefix="movis_audio_track")
        clip_duration = _safe_float(duration, 0.0, min_value=0.0)
        if clip_duration <= 0:
            clip_duration = _audio_duration(path)
        t["audio_tracks"].append(
            {
                "path": path,
                "start": _safe_float(start, 0.0, min_value=0.0),
                "duration": max(0.01, clip_duration),
                "source_start": _safe_float(source_start, 0.0, min_value=0.0),
                "audio_level_db": _safe_float(audio_level_db, 0.0, min_value=-60.0, max_value=24.0),
            }
        )
        return (t,)


class MovisSetBGM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "bgm_path": ("STRING", {"default": "", "placeholder": "可留空，改接 AUDIO"}),
                "audio_level_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 24.0}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_bgm"

    def set_bgm(self, timeline, bgm_path, audio_level_db, source_start, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_audio_input(bgm_path, audio=audio, prefix="movis_bgm")
        t["bgm"] = {
            "path": path,
            "audio_level_db": _safe_float(audio_level_db, -12.0, min_value=-60.0, max_value=24.0),
            "source_start": _safe_float(source_start, 0.0, min_value=0.0),
            "duration": _audio_duration(path),
        }
        return (t,)


class MovisAddTextOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "text": ("STRING", {"default": "字幕文本", "multiline": True}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "duration": ("FLOAT", {"default": 2.0, "min": 0.01}),
                "font_size": ("FLOAT", {"default": 64.0, "min": 8.0, "max": 300.0}),
                "font_family": ("STRING", {"default": "Sans Serif"}),
                "color": ("STRING", {"default": "white"}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "position_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_text"

    def add_text(self, timeline, text, start, duration, font_size, font_family, color, position_x, position_y, opacity):
        t = _clone_timeline(timeline)
        t["text_tracks"].append(
            {
                "text": text,
                "start": _safe_float(start, 0.0, min_value=0.0),
                "duration": _safe_float(duration, 2.0, min_value=0.01),
                "font_size": _safe_float(font_size, 64.0, min_value=8.0, max_value=300.0),
                "font_family": font_family,
                "color": color,
                "position_x": _safe_float(position_x, 0.5),
                "position_y": _safe_float(position_y, 0.9),
                "opacity": _safe_float(opacity, 1.0, min_value=0.0, max_value=1.0),
            }
        )
        return (t,)


class MovisRenderTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "output_file_prefix": ("STRING", {"default": "movis_pipeline_", "tooltip": "输出文件名前缀"}),
                "notify_all": ("BOOLEAN", {"default": False, "tooltip": "渲染完成后发送通知"}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "audio", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "render"
    OUTPUT_NODE = True

    def render(self, timeline, output_file_prefix, notify_all):
        out_path, audio, frames, duration = _render_timeline(timeline, output_file_prefix, notify_all)
        return {"ui": {"video_path": out_path, "frames": [frames]}, "result": (out_path, audio, frames, duration)}


class MovisAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("STRING", {"default": "", "placeholder": "路径列表；留空可改接 VHS_FILENAMES"}),
                "audios": ("STRING", {"default": "", "placeholder": "路径列表；留空可改接 AUDIO"}),
                "bgm": ("STRING", {"default": "", "placeholder": "路径；留空可改接 AUDIO"}),
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "crossfade": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "output_file_prefix": ("STRING", {"default": "movis_assemble_"}),
                "notify_all": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
                "audio": ("AUDIO",),
                "bgm_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "audio", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "assemble"
    OUTPUT_NODE = True

    def assemble(self, videos, audios, bgm, width, height, fps, crossfade, output_file_prefix, notify_all, vhs_filenames=None, audio=None, bgm_audio=None):
        timeline = _new_timeline(width, height, fps, "#000000")
        video_paths = [resolve_media_path(p, must_exist=True) for p in _split_paths(videos)]
        if len(video_paths) == 0:
            video_paths = _extract_video_paths_from_vhs_filenames(vhs_filenames)

        audio_paths = [resolve_media_path(p, must_exist=True) for p in _split_paths(audios)]
        if len(audio_paths) == 0 and audio is not None:
            audio_paths = [_audio_to_wav_file(audio, prefix="movis_assemble_audio")]
        cursor = 0.0
        for idx, vpath in enumerate(video_paths):
            dur = _video_duration(vpath)
            timeline["video_tracks"].append(
                {
                    "path": vpath,
                    "is_image": False,
                    "start": cursor,
                    "duration": dur,
                    "source_start": 0.0,
                    "fade_in": crossfade if idx > 0 else 0.0,
                    "fade_out": crossfade if idx < len(video_paths) - 1 else 0.0,
                    "use_source_audio": idx >= len(audio_paths),
                    "audio_level_db": 0.0,
                    "position_x": 0.5,
                    "position_y": 0.5,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "rotation": 0.0,
                    "opacity": 1.0,
                }
            )
            if idx < len(audio_paths):
                adur = _audio_duration(audio_paths[idx])
                timeline["audio_tracks"].append(
                    {
                        "path": audio_paths[idx],
                        "start": cursor,
                        "duration": adur,
                        "source_start": 0.0,
                        "audio_level_db": 0.0,
                    }
                )
                dur = adur
            cursor += max(0.01, dur - crossfade)

        if bgm or bgm_audio is not None:
            bgm_path = _resolve_audio_input(bgm, audio=bgm_audio, prefix="movis_assemble_bgm")
            timeline["bgm"] = {
                "path": bgm_path,
                "audio_level_db": -12.0,
                "source_start": 0.0,
                "duration": _audio_duration(bgm_path),
            }

        out_path, audio, frames, duration = _render_timeline(timeline, output_file_prefix, notify_all)
        return {"ui": {"video_path": out_path, "frames": [frames]}, "result": (out_path, audio, frames, duration)}


class MovisUniversalStudio:
    """一体化专业节点：多视频+多音频+可选BGM，支持 concat/overlay/custom_start 三种编排。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("STRING", {"default": "", "multiline": True, "placeholder": "视频路径列表：支持逗号/分号/换行", "tooltip": "可直接填多个视频路径；为空时可改接 VHS_FILENAMES"}),
                "audios": ("STRING", {"default": "", "multiline": True, "placeholder": "外部音频路径列表（可空）"}),
                "bgm": ("STRING", {"default": "", "placeholder": "BGM 路径，可空"}),
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "compose_mode": (["concat", "overlay", "custom_start"], {"default": "concat", "tooltip": "concat: 顺序拼接；overlay: 全部从0秒叠加；custom_start: 按 start_csv"}),
                "video_starts_csv": ("STRING", {"default": "", "placeholder": "仅 custom_start 使用，如 0,2.5,8"}),
                "video_durations_csv": ("STRING", {"default": "", "placeholder": "每段时长；<=0 自动读取源时长"}),
                "video_source_starts_csv": ("STRING", {"default": "", "placeholder": "每段源起点，如 0,1.2,0"}),
                "video_use_source_audio_csv": ("STRING", {"default": "", "placeholder": "如 true,false,true"}),
                "video_audio_level_csv": ("STRING", {"default": "", "placeholder": "每段视频自带音轨增益(dB)"}),
                "audio_starts_csv": ("STRING", {"default": "", "placeholder": "外部音频开始时间列表"}),
                "audio_durations_csv": ("STRING", {"default": "", "placeholder": "外部音频时长列表；<=0 自动读取"}),
                "audio_source_starts_csv": ("STRING", {"default": "", "placeholder": "外部音频源起点列表"}),
                "audio_level_csv": ("STRING", {"default": "", "placeholder": "外部音频增益(dB)列表"}),
                "crossfade": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 8.0, "tooltip": "concat 模式的视频交叉淡化时长"}),
                "bgm_level_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 24.0}),
                "auto_duck_bgm": ("BOOLEAN", {"default": True, "tooltip": "有前景音频时自动压低 BGM"}),
                "duck_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 12.0}),
                "output_file_prefix": ("STRING", {"default": "movis_studio_"}),
                "notify_all": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
                "audio": ("AUDIO",),
                "bgm_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO", "INT", "FLOAT", "MOVIS_TIMELINE")
    RETURN_NAMES = ("video_path", "audio", "frames", "duration", "timeline")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "studio"
    OUTPUT_NODE = True

    def studio(
        self,
        videos,
        audios,
        bgm,
        width,
        height,
        fps,
        compose_mode,
        video_starts_csv,
        video_durations_csv,
        video_source_starts_csv,
        video_use_source_audio_csv,
        video_audio_level_csv,
        audio_starts_csv,
        audio_durations_csv,
        audio_source_starts_csv,
        audio_level_csv,
        crossfade,
        bgm_level_db,
        auto_duck_bgm,
        duck_db,
        output_file_prefix,
        notify_all,
        vhs_filenames=None,
        audio=None,
        bgm_audio=None,
    ):
        timeline = _new_timeline(
            int(_safe_float(width, 1080.0, min_value=64.0, max_value=8192.0)),
            int(_safe_float(height, 1920.0, min_value=64.0, max_value=8192.0)),
            _safe_float(fps, 30.0, min_value=1.0, max_value=240.0),
            "#000000",
        )

        video_paths = [resolve_media_path(p, must_exist=True) for p in _split_paths(videos)]
        if len(video_paths) == 0:
            video_paths = _extract_video_paths_from_vhs_filenames(vhs_filenames)

        if len(video_paths) == 0:
            raise ValueError("未提供有效视频输入：请设置 videos 或连接 VHS_FILENAMES")

        ext_audio_paths = [resolve_media_path(p, must_exist=True) for p in _split_paths(audios)]
        if len(ext_audio_paths) == 0 and audio is not None:
            ext_audio_paths = [_audio_to_wav_file(audio, prefix="movis_studio_audio")]

        starts = _parse_float_list(video_starts_csv)
        durs = _parse_float_list(video_durations_csv)
        src_starts = _parse_float_list(video_source_starts_csv)
        use_src_audio = _parse_bool_list(video_use_source_audio_csv)
        video_audio_levels = _parse_float_list(video_audio_level_csv)

        cursor = 0.0
        cf = _safe_float(crossfade, 0.35, min_value=0.0, max_value=8.0)
        for idx, vpath in enumerate(video_paths):
            src_start = src_starts[idx] if idx < len(src_starts) else 0.0
            src_start = _safe_float(src_start, 0.0, min_value=0.0)

            vdur = durs[idx] if idx < len(durs) else 0.0
            vdur = _safe_float(vdur, 0.0)
            if vdur <= 0:
                vdur = _video_duration(vpath)

            if compose_mode == "overlay":
                start = 0.0
            elif compose_mode == "custom_start":
                start = _safe_float(starts[idx] if idx < len(starts) else 0.0, 0.0, min_value=0.0)
            else:
                start = cursor

            use_v_audio = use_src_audio[idx] if idx < len(use_src_audio) else (idx >= len(ext_audio_paths))
            v_audio_db = _safe_float(video_audio_levels[idx] if idx < len(video_audio_levels) else 0.0, 0.0, min_value=-60.0, max_value=24.0)

            timeline["video_tracks"].append(
                {
                    "path": vpath,
                    "is_image": False,
                    "start": start,
                    "duration": max(0.01, vdur),
                    "source_start": src_start,
                    "fade_in": cf if compose_mode == "concat" and idx > 0 else 0.0,
                    "fade_out": cf if compose_mode == "concat" and idx < len(video_paths) - 1 else 0.0,
                    "use_source_audio": bool(use_v_audio),
                    "audio_level_db": v_audio_db,
                    "position_x": 0.5,
                    "position_y": 0.5,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "rotation": 0.0,
                    "opacity": 1.0,
                }
            )

            if compose_mode == "concat":
                cursor += max(0.01, vdur - cf)

        a_starts = _parse_float_list(audio_starts_csv)
        a_durs = _parse_float_list(audio_durations_csv)
        a_src_starts = _parse_float_list(audio_source_starts_csv)
        a_levels = _parse_float_list(audio_level_csv)
        for idx, apath in enumerate(ext_audio_paths):
            adur = _safe_float(a_durs[idx] if idx < len(a_durs) else 0.0, 0.0)
            if adur <= 0:
                adur = _audio_duration(apath)
            timeline["audio_tracks"].append(
                {
                    "path": apath,
                    "start": _safe_float(a_starts[idx] if idx < len(a_starts) else 0.0, 0.0, min_value=0.0),
                    "duration": max(0.01, adur),
                    "source_start": _safe_float(a_src_starts[idx] if idx < len(a_src_starts) else 0.0, 0.0, min_value=0.0),
                    "audio_level_db": _safe_float(a_levels[idx] if idx < len(a_levels) else 0.0, 0.0, min_value=-60.0, max_value=24.0),
                }
            )

        if bgm or bgm_audio is not None:
            bgm_path = _resolve_audio_input(bgm, audio=bgm_audio, prefix="movis_studio_bgm")
            timeline["bgm"] = {
                "path": bgm_path,
                "audio_level_db": _safe_float(bgm_level_db, -12.0, min_value=-60.0, max_value=24.0),
                "source_start": 0.0,
                "duration": _audio_duration(bgm_path),
            }

        if auto_duck_bgm and timeline.get("bgm"):
            timeline["bgm"]["audio_level_db"] = _smart_bgm_level(
                timeline,
                default_bgm_db=float(timeline["bgm"].get("audio_level_db", -12.0)),
                duck_db=_safe_float(duck_db, -18.0, min_value=-60.0, max_value=12.0),
            )

        out_path, out_audio, frames, duration = _render_timeline(timeline, output_file_prefix, notify_all)
        return {
            "ui": {"video_path": out_path, "frames": [frames]},
            "result": (out_path, out_audio, frames, duration, timeline),
        }


class MovisSmartMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_a": ("MOVIS_TIMELINE",),
                "timeline_b": ("MOVIS_TIMELINE",),
                "mode": (["overlay", "concat", "smart"], {"default": "smart", "tooltip": "smart会自动根据时长与内容决定拼接或叠加"}),
                "gap_seconds": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "tooltip": "concat模式下B相对A结束时间的偏移"}),
                "merge_bgm": (["keep_a", "use_b", "mix_as_audio_track", "smart"], {"default": "smart"}),
                "auto_duck_bgm": ("BOOLEAN", {"default": True, "tooltip": "检测到前景音频时自动压低BGM"}),
                "duck_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 12.0}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("timeline", "offset_b", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "smart_merge"

    def smart_merge(self, timeline_a, timeline_b, mode, gap_seconds, merge_bgm, auto_duck_bgm, duck_db):
        ta = _clone_timeline(timeline_a)
        tb = _clone_timeline(timeline_b)
        da = _timeline_duration(ta)
        db = _timeline_duration(tb)

        if mode == "overlay":
            offset_b = 0.0
        elif mode == "concat":
            offset_b = float(da + gap_seconds)
        else:
            # smart: 若A已有较完整时长内容，优先串联；否则叠加
            offset_b = float(da + gap_seconds) if da > 1.0 and db > 1.0 else 0.0

        t = _offset_timeline(tb, offset_b)
        ta["video_tracks"].extend(t.get("video_tracks", []))
        ta["audio_tracks"].extend(t.get("audio_tracks", []))
        ta["text_tracks"].extend(t.get("text_tracks", []))

        bgm_b = t.get("bgm")
        if merge_bgm == "use_b":
            ta["bgm"] = bgm_b
        elif merge_bgm == "mix_as_audio_track" and bgm_b and bgm_b.get("path"):
            ta["audio_tracks"].append(
                {
                    "path": bgm_b["path"],
                    "start": float(offset_b),
                    "duration": float(bgm_b.get("duration", 0.0) or 0.0),
                    "source_start": float(bgm_b.get("source_start", 0.0)),
                    "audio_level_db": float(bgm_b.get("audio_level_db", -12.0)),
                }
            )
        elif merge_bgm == "smart":
            bgm_a = ta.get("bgm")
            if bgm_a and bgm_a.get("path"):
                pass
            elif bgm_b and bgm_b.get("path"):
                ta["bgm"] = bgm_b

        if auto_duck_bgm and ta.get("bgm"):
            ta["bgm"]["audio_level_db"] = _smart_bgm_level(ta, default_bgm_db=float(ta["bgm"].get("audio_level_db", -12.0)), duck_db=float(duck_db))

        return (ta, float(offset_b), float(_timeline_duration(ta)))


class MovisTimelinePro:
    """兼容入口：内部转换为新的可管线化结构。"""

    @classmethod
    def INPUT_TYPES(cls):
        example = '{"canvas":{"width":1080,"height":1920,"fps":30},"video_tracks":[],"audio_tracks":[],"bgm":{"path":"","audio_level_db":-12}}'
        return {
            "required": {
                "timeline_json": ("STRING", {"default": example, "multiline": True}),
                "output_file_prefix": ("STRING", {"default": "movis_timeline_pro_"}),
                "notify_all": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("video_file_path", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "render"
    OUTPUT_NODE = True

    def render(self, timeline_json: str, output_file_prefix: str, notify_all: bool):
        import json

        data = json.loads(timeline_json)
        canvas = data.get("canvas", {})
        timeline = _new_timeline(
            int(canvas.get("width", 1080)),
            int(canvas.get("height", 1920)),
            float(canvas.get("fps", 30.0)),
            str(canvas.get("bg_color", "#000000")),
        )
        for clip in data.get("video_tracks", []) or []:
            path = resolve_media_path(clip.get("path", ""), must_exist=True)
            is_image = _is_image(path)
            dur = float(clip.get("duration", 0) or 0)
            if dur <= 0:
                dur = _video_duration(path) if not is_image else float(clip.get("image_duration", 3.0))
            timeline["video_tracks"].append(
                {
                    "path": path,
                    "is_image": is_image,
                    "start": float(clip.get("start", 0)),
                    "duration": max(0.01, dur),
                    "source_start": float(clip.get("source_start", 0)),
                    "fade_in": float(clip.get("crossfade_in", 0)),
                    "fade_out": float(clip.get("crossfade_out", 0)),
                    "use_source_audio": bool(clip.get("use_source_audio", True)),
                    "audio_level_db": float(clip.get("audio_level_db", 0)),
                    "position_x": 0.5,
                    "position_y": 0.5,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "rotation": 0.0,
                    "opacity": 1.0,
                }
            )

        for track in data.get("audio_tracks", []) or []:
            ap = resolve_media_path(track.get("path", ""), must_exist=True)
            dur = float(track.get("duration", 0) or 0)
            if dur <= 0:
                dur = _audio_duration(ap)
            timeline["audio_tracks"].append(
                {
                    "path": ap,
                    "start": float(track.get("start", 0)),
                    "duration": max(0.01, dur),
                    "source_start": float(track.get("source_start", 0)),
                    "audio_level_db": float(track.get("audio_level_db", 0)),
                }
            )

        bgm = (data.get("bgm") or {})
        if bgm.get("path"):
            bp = resolve_media_path(bgm.get("path"), must_exist=True)
            timeline["bgm"] = {
                "path": bp,
                "audio_level_db": float(bgm.get("audio_level_db", -12)),
                "source_start": float(bgm.get("source_start", 0)),
                "duration": _audio_duration(bp),
            }

        out_path, _audio, frames, duration = _render_timeline(timeline, output_file_prefix, notify_all)
        return {"ui": {"video_path": out_path, "frames": [frames]}, "result": (out_path, frames, duration)}


class CompositeMedia:
    """Deprecated：兼容旧 flow。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audios": ("STRING", {"default": ""}),
                "is_vertical": ("BOOLEAN", {"default": True}),
                "output_file_prefix": ("STRING", {"default": "composite_output_"}),
                "notify_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {"bgm": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_file_path",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "composite_media"
    OUTPUT_NODE = True

    def composite_media(self, images, audios, bgm, is_vertical, output_file_prefix, notify_all):
        width, height = (1080, 1920) if is_vertical else (1920, 1080)
        timeline = _new_timeline(width, height, 30.0, "#000000")
        img_files = _save_images(images, f"legacy_media_{_now_stamp()}")
        audio_files = [resolve_media_path(p, must_exist=True) for p in _split_paths(audios)]
        cursor = 0.0
        for idx, (img, ap) in enumerate(zip(img_files, audio_files)):
            dur = _audio_duration(ap)
            timeline["video_tracks"].append(
                {
                    "path": img,
                    "is_image": True,
                    "start": cursor,
                    "duration": dur,
                    "source_start": 0.0,
                    "fade_in": 0.35 if idx > 0 else 0.0,
                    "fade_out": 0.35 if idx < len(audio_files) - 1 else 0.0,
                    "use_source_audio": False,
                    "audio_level_db": 0.0,
                    "position_x": 0.5,
                    "position_y": 0.5,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "rotation": 0.0,
                    "opacity": 1.0,
                }
            )
            timeline["audio_tracks"].append(
                {
                    "path": ap,
                    "start": cursor,
                    "duration": dur,
                    "source_start": 0.0,
                    "audio_level_db": 0.0,
                }
            )
            cursor += max(0.01, dur - 0.35)

        if bgm:
            bp = resolve_media_path(bgm, must_exist=True)
            timeline["bgm"] = {
                "path": bp,
                "audio_level_db": -12.0,
                "source_start": 0.0,
                "duration": _audio_duration(bp),
            }

        out_path, _audio, _frames, _duration = _render_timeline(timeline, output_file_prefix, notify_all)
        return {"ui": {"video_path": out_path}, "result": (out_path,)}


class CompositeMultiVideo:
    """Deprecated：兼容旧 flow。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("STRING", {"default": ""}),
                "audios": ("STRING", {"default": ""}),
                "is_vertical": ("BOOLEAN", {"default": True}),
                "output_file_prefix": ("STRING", {"default": "composite_multi_video_output_"}),
                "notify_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {"bgm": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_file_path",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "composite_videos"
    OUTPUT_NODE = True

    def composite_videos(self, videos, audios, bgm, is_vertical, output_file_prefix, notify_all):
        width, height = (1080, 1920) if is_vertical else (1920, 1080)
        result = MovisAssemble().assemble(videos, audios, bgm, width, height, 30.0, 0.35, output_file_prefix, notify_all)
        return {"ui": result["ui"], "result": (result["result"][0],)}
