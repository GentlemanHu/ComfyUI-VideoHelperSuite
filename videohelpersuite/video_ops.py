import copy
import base64
import datetime
import difflib
import io
import os
import re
import sys
import subprocess
import wave
from pathlib import Path
from typing import Any


def _is_truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_qt_runtime_for_movis() -> None:
    """安全配置 Qt 运行时（仅在必要场景启用无头参数）。"""
    if not sys.platform.startswith("linux"):
        return

    force = _is_truthy_env("COMFYUI_FORCE_HEADLESS_QT")
    has_display = any(
        bool(str(os.environ.get(k, "")).strip())
        for k in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    )

    plugin_path = str(os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""))
    if "cv2/qt/plugins" in plugin_path.replace("\\", "/").lower():
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    if not force and has_display:
        return

    current_platform = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
    if current_platform in {"", "xcb"}:
        os.environ["QT_QPA_PLATFORM"] = str(
            os.environ.get("COMFYUI_HEADLESS_QT_PLATFORM", "offscreen")
        ).strip() or "offscreen"

    os.environ.setdefault("QT_OPENGL", "software")


configure_qt_runtime_for_movis()

import movis as mv
import numpy as np
import torch
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from comfy.utils import ProgressBar
from tqdm import tqdm

import folder_paths

from .caption import resolve_media_path
from .utils import ffmpeg_path, get_audio

try:
    from notifier.notify import notifyAll
except Exception:
    notifyAll = None


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


_FONT_CACHE = None
_STYLE_CACHE = {}
_FONT_CACHE_VERSION = 0
_TEXT_SUPPORTS_FONT_STYLE = None
_COMMON_FONT_STYLES = ["Regular", "Bold", "Italic", "Bold Italic", "Light", "Medium", "SemiBold"]


def get_movis_fonts(refresh: bool = False) -> list[str]:
    global _FONT_CACHE, _FONT_CACHE_VERSION
    if _FONT_CACHE is not None and not refresh:
        return list(_FONT_CACHE)

    configure_qt_runtime_for_movis()
    try:
        fonts = sorted(set(str(x).strip() for x in mv.layer.Text.available_fonts() if str(x).strip()))
        if not fonts:
            raise RuntimeError("movis returned empty font list")
        _FONT_CACHE = fonts
    except Exception as e:
        print(f"[Movis Font] Failed to enumerate fonts via movis/Qt: {e}")
        _FONT_CACHE = [
            "Sans Serif",
            "Arial",
            "Helvetica",
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "PingFang SC",
            "Lemon",
        ]

    _FONT_CACHE_VERSION += 1
    return list(_FONT_CACHE)


def get_movis_font_cache_version() -> int:
    return int(_FONT_CACHE_VERSION)


def clear_movis_font_caches() -> None:
    global _FONT_CACHE, _STYLE_CACHE
    _FONT_CACHE = None
    _STYLE_CACHE.clear()


def get_movis_styles(font_family: str, refresh: bool = False) -> list[str]:
    global _STYLE_CACHE
    family = str(font_family or "").strip()
    if not family:
        return ["Regular"]
    if (not refresh) and family in _STYLE_CACHE:
        return list(_STYLE_CACHE[family])

    configure_qt_runtime_for_movis()
    try:
        styles = sorted(set(str(x).strip() for x in mv.layer.Text.available_styles(family) if str(x).strip()))
        if not styles:
            styles = ["Regular"]
    except Exception as e:
        print(f"[Movis Font] Failed to enumerate styles for {family}: {e}")
        styles = ["Regular"]
    _STYLE_CACHE[family] = styles
    return list(styles)


def choose_default_font(fonts: list[str]) -> str:
    preferred = [
        "Lemon",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "PingFang SC",
        "Sans Serif",
        "Arial",
        "Helvetica",
    ]
    for name in preferred:
        if name in fonts:
            return name
    return fonts[0] if fonts else "Sans Serif"


def validate_font_family(font_family: str) -> str:
    fonts = get_movis_fonts()
    family = str(font_family or "").strip()
    if family in fonts:
        return family

    default = choose_default_font(fonts)
    close = difflib.get_close_matches(family, fonts, n=5)
    print(
        f"[Movis Font] Font not found: {family!r}. "
        f"Close matches: {close}. Fallback: {default!r}"
    )
    return default


def validate_font_style(font_family: str, font_style: str) -> str:
    styles = get_movis_styles(font_family)
    style = str(font_style or "").strip()
    if style in styles:
        return style

    fallback = "Regular" if "Regular" in styles else (styles[0] if styles else "Regular")
    print(
        f"[Movis Font] Style not found: {style!r} for {font_family!r}. "
        f"Available styles: {styles}. Fallback: {fallback!r}"
    )
    return fallback


def normalize_movis_font_request(font_family: str, font_style: str = "Regular") -> tuple[str, str]:
    family = str(font_family or "").strip()
    style = str(font_style or "").strip() or "Regular"

    if not family:
        safe_family = validate_font_family(choose_default_font(get_movis_fonts()))
        safe_style = validate_font_style(safe_family, style)
        return safe_family, safe_style

    fonts = get_movis_fonts()
    if family in fonts:
        return family, validate_font_style(family, style)

    family_compact = family.replace("_", " ").strip()
    if family_compact in fonts:
        return family_compact, validate_font_style(family_compact, style)

    # 兼容旧工作流常见写法：Lemon-Regular / UbuntuMono-Bold / Roboto Bold
    split_candidates = []
    for sep in ("-", ",", "|"):
        if sep in family:
            left, right = family.rsplit(sep, 1)
            split_candidates.append((left.strip(), right.strip()))
    parts = family.split()
    if len(parts) >= 2:
        split_candidates.append((" ".join(parts[:-1]).strip(), parts[-1].strip()))

    for maybe_family, maybe_style in split_candidates:
        if not maybe_family:
            continue
        safe_family = validate_font_family(maybe_family)
        family_styles = get_movis_styles(safe_family)
        candidate_style = str(maybe_style or style).strip() or style
        if candidate_style in family_styles:
            return safe_family, candidate_style
        if style in family_styles:
            return safe_family, style
        if candidate_style:
            lowered = {s.lower(): s for s in family_styles}
            if candidate_style.lower() in lowered:
                return safe_family, lowered[candidate_style.lower()]

    safe_family = validate_font_family(family)
    safe_style = validate_font_style(safe_family, style)
    return safe_family, safe_style


def _movis_text_supports_font_style() -> bool:
    global _TEXT_SUPPORTS_FONT_STYLE
    if _TEXT_SUPPORTS_FONT_STYLE is not None:
        return bool(_TEXT_SUPPORTS_FONT_STYLE)
    try:
        import inspect
        sig = inspect.signature(mv.layer.Text)
        _TEXT_SUPPORTS_FONT_STYLE = "font_style" in sig.parameters
    except Exception:
        _TEXT_SUPPORTS_FONT_STYLE = False
    return bool(_TEXT_SUPPORTS_FONT_STYLE)


def _create_movis_text_layer(text: str, font_size: float, font_family: str, font_style: str, color: str, duration: float):
    kwargs = {
        "text": str(text),
        "font_size": float(font_size),
        "font_family": str(font_family),
        "color": str(color),
        "duration": float(duration),
    }
    if _movis_text_supports_font_style():
        kwargs["font_style"] = str(font_style)
    try:
        return mv.layer.Text(**kwargs)
    except TypeError:
        kwargs.pop("font_style", None)
        return mv.layer.Text(**kwargs)


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    return int(round(_safe_float(value, float(default), min_value=float(min_value), max_value=float(max_value))))


def _safe_color_text(value: str, default: str) -> str:
    try:
        ImageColor.getcolor(str(value), "RGBA")
        return str(value)
    except Exception:
        return str(default)


def _font_scan_dirs() -> list[str]:
    if sys.platform == "win32":
        return [
            "C:/Windows/Fonts",
            str((Path.home() / "AppData/Local/Microsoft/Windows/Fonts").resolve()),
        ]
    if sys.platform == "darwin":
        return [
            "/System/Library/Fonts",
            "/Library/Fonts",
            str((Path.home() / "Library/Fonts").resolve()),
        ]
    return [
        "/root/.local/share/fonts/custom",
        str((Path.home() / ".local/share/fonts").resolve()),
        str((Path.home() / ".fonts").resolve()),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]


def _font_contains_text(font_path: str, text: str) -> tuple[bool, list[str], int]:
    chars = [ch for ch in str(text or "") if not ch.isspace()]
    if not chars:
        return True, [], 0
    try:
        from fontTools.ttLib import TTCollection, TTFont  # lazy import

        cmap = {}
        if str(font_path).lower().endswith(".ttc"):
            collection = TTCollection(font_path)
            for ft in collection.fonts:
                for table in ft["cmap"].tables:
                    cmap.update(table.cmap)
        else:
            font = TTFont(font_path, fontNumber=0)
            for table in font["cmap"].tables:
                cmap.update(table.cmap)

        missing = [ch for ch in chars if ord(ch) not in cmap]
        return len(missing) == 0, missing, len(cmap)
    except ImportError:
        print("[Movis Font Preview] Warning: fontTools 未安装，跳过字符覆盖检测。")
        return True, [], -1
    except Exception as e:
        print(f"[Movis Font Preview] Warning: 字体覆盖检测失败: {font_path!r}, error={e}")
        return True, [], -1


def font_support_report(font_path: str, text: str) -> dict[str, Any]:
    supports_text, missing_chars, cmap_count = _font_contains_text(font_path, text)
    return {
        "supports_text": bool(supports_text),
        "missing_chars": missing_chars,
        "cmap_count": int(cmap_count),
    }


def _append_font_candidate(candidates: list[str], path: str) -> None:
    p = str(path or "").strip()
    if p and os.path.isfile(p):
        candidates.append(p)


def _font_name_aliases(font_path: str) -> set[str]:
    aliases: set[str] = set()
    try:
        from fontTools.ttLib import TTCollection, TTFont

        fonts = []
        if str(font_path).lower().endswith(".ttc"):
            fonts = list(TTCollection(font_path).fonts)
        else:
            fonts = [TTFont(font_path, fontNumber=0)]

        for f in fonts:
            name_table = f.get("name")
            if not name_table:
                continue
            for rec in name_table.names:
                try:
                    text = rec.toUnicode().strip()
                except Exception:
                    continue
                if text:
                    aliases.add(text.lower().replace(" ", "").replace("-", ""))
    except Exception:
        pass
    return aliases


def _find_font_file_for_preview(font_family: str, font_style: str, preview_text: str) -> tuple[str | None, dict[str, Any]]:
    family = str(font_family or "").strip()
    style = str(font_style or "").strip()
    if not family:
        return None, {"supports_text": False, "missing_chars": [], "cmap_count": 0}

    candidates: list[str] = []

    if sys.platform.startswith("linux"):
        if style:
            try:
                ret = subprocess.run(
                    ["fc-match", f"{family}:style={style}", "-f", "%{file}"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                if ret.returncode == 0:
                    _append_font_candidate(candidates, str(ret.stdout or "").strip())
            except Exception:
                pass
        try:
            ret = subprocess.run(
                ["fc-match", family, "-f", "%{file}"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if ret.returncode == 0:
                _append_font_candidate(candidates, str(ret.stdout or "").strip())
        except Exception:
            pass

    try:
        from matplotlib import font_manager  # lazy import
        _append_font_candidate(candidates, font_manager.findfont(family, fallback_to_default=True))
    except Exception:
        pass

    family_compact = family.lower().replace(" ", "").replace("-", "")
    style_compact = style.lower().replace(" ", "").replace("-", "")
    for base in _font_scan_dirs():
        try:
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for f in files:
                    fl = f.lower()
                    if not fl.endswith((".ttf", ".otf", ".ttc")):
                        continue
                    font_path = os.path.join(root, f)
                    compact = fl.replace(" ", "").replace("-", "")
                    aliases = _font_name_aliases(font_path)
                    family_ok = (family_compact in compact) or any(family_compact in a for a in aliases)
                    if family_compact and not family_ok:
                        continue
                    if style_compact and style_compact not in compact and style_compact not in {"regular", "normal"}:
                        style_ok = any(style_compact in a for a in aliases)
                        if not style_ok:
                            continue
                    _append_font_candidate(candidates, font_path)
        except Exception:
            continue

    dedup_candidates = list(dict.fromkeys(candidates))
    fallback_path = dedup_candidates[0] if dedup_candidates else None
    for p in dedup_candidates:
        report = font_support_report(p, preview_text)
        if bool(report.get("supports_text", False)):
            return p, report

    if fallback_path:
        report = font_support_report(fallback_path, preview_text)
        print(
            f"[Movis Font Preview] Warning: family={family!r}, style={style!r} "
            f"未找到完整支持文本的字体，回退使用: {fallback_path!r}, "
            f"missing_chars={report.get('missing_chars', [])[:32]}"
        )
        return fallback_path, report

    print(f"[Movis Font Preview] Warning: 未找到可用字体文件 family={family!r}, style={style!r}")
    return None, {"supports_text": False, "missing_chars": [], "cmap_count": 0}


def render_font_preview_image(
    font_family: str,
    font_style: str,
    text: str,
    font_size: int,
    width: int,
    height: int,
    text_color: str,
    bg_color: str,
    x: float = 0.5,
    y: float = 0.5,
    align: str = "center",
    stroke_width: int = 0,
    stroke_color: str = "#000000",
) -> tuple[Image.Image, dict[str, Any]]:
    safe_family = validate_font_family(font_family)
    safe_style = validate_font_style(safe_family, font_style)
    safe_text_color = _safe_color_text(text_color, "#ffffff")
    safe_bg_color = _safe_color_text(bg_color, "#222222")
    safe_stroke_color = _safe_color_text(stroke_color, "#000000")

    w = _clamp_int(width, 768, 64, 4096)
    h = _clamp_int(height, 200, 64, 2048)
    fs = _clamp_int(font_size, 64, 8, 512)
    sw = _clamp_int(stroke_width, 0, 0, 128)
    anchor_x = _safe_float(x, 0.5)
    anchor_y = _safe_float(y, 0.5)
    safe_align = str(align or "center").strip().lower()
    if safe_align not in {"left", "center", "right"}:
        safe_align = "center"

    text_value = str(text or "")
    font_path, report = _find_font_file_for_preview(safe_family, safe_style, text_value)
    font = ImageFont.load_default()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=fs)
        except Exception as e:
            print(f"[Movis Font Preview] Warning: 字体加载失败 {font_path!r}, error={e}")
    else:
        print(f"[Movis Font Preview] Warning: 预览将使用默认字体 family={safe_family!r}, style={safe_style!r}")

    supports_text = bool(report.get("supports_text", False))
    if os.environ.get("VHS_MOVIS_FONT_PREVIEW_LOG", "1").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[Movis Font Preview] family={safe_family!r}, style={safe_style!r}, path={font_path!r}")
        print(f"[Movis Font Preview] text={text_value!r}, supports_text={supports_text}")

    image = Image.new("RGBA", (w, h), safe_bg_color)
    draw = ImageDraw.Draw(image)

    spacing = 4
    pad = max(8, int(round(fs * 0.15)) + sw)
    avail_w = max(1, w - pad * 2)
    avail_h = max(1, h - pad * 2)

    # 预览避免“文字被截断”：自动缩放字体直到放进可视区域。
    fit_font = font
    fit_bbox = draw.multiline_textbbox(
        (0, 0),
        text_value,
        font=fit_font,
        align=safe_align,
        stroke_width=sw,
        spacing=spacing,
    )
    fit_size = fs
    if font_path:
        for try_size in range(fs, 7, -2):
            try:
                test_font = ImageFont.truetype(font_path, size=try_size)
            except Exception:
                break
            test_bbox = draw.multiline_textbbox(
                (0, 0),
                text_value,
                font=test_font,
                align=safe_align,
                stroke_width=sw,
                spacing=spacing,
            )
            test_w = max(1, int(test_bbox[2] - test_bbox[0]))
            test_h = max(1, int(test_bbox[3] - test_bbox[1]))
            if test_w <= avail_w and test_h <= avail_h:
                fit_font = test_font
                fit_bbox = test_bbox
                fit_size = try_size
                break

    text_w = max(1, int(fit_bbox[2] - fit_bbox[0]))
    text_h = max(1, int(fit_bbox[3] - fit_bbox[1]))

    px = _position_value_to_pixels(anchor_x, w)
    py = _position_value_to_pixels(anchor_y, h)
    if safe_align == "left":
        draw_x = px
    elif safe_align == "right":
        draw_x = px - text_w
    else:
        draw_x = px - text_w * 0.5
    draw_y = py - text_h * 0.5

    # 使用 bbox 偏移做边界保护，防止 stroke/字形外扩导致被裁切。
    min_x = pad - float(fit_bbox[0])
    max_x = (w - pad) - float(fit_bbox[2])
    min_y = pad - float(fit_bbox[1])
    max_y = (h - pad) - float(fit_bbox[3])
    if min_x <= max_x:
        draw_x = max(min_x, min(draw_x, max_x))
    if min_y <= max_y:
        draw_y = max(min_y, min(draw_y, max_y))

    draw.multiline_text(
        (float(draw_x), float(draw_y)),
        text_value,
        font=fit_font,
        fill=safe_text_color,
        align=safe_align,
        stroke_width=sw,
        stroke_fill=safe_stroke_color,
        spacing=spacing,
    )

    meta = {
        "font_path": font_path,
        "supports_text": supports_text,
        "missing_chars": list(report.get("missing_chars", [])),
        "cmap_count": int(report.get("cmap_count", 0)),
        "family": safe_family,
        "style": safe_style,
        "fit_font_size": int(fit_size),
        "draw_padding": int(pad),
    }
    return image, meta


def encode_preview_image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _pil_to_comfy_image(img: Image.Image):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _check_nvenc_available() -> bool:
    """Probe whether h264_nvenc is usable on this ffmpeg build."""
    ff = ffmpeg_path or "ffmpeg"
    try:
        r = subprocess.run(
            [ff, "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "16x16", "-r", "1", "-i", "pipe:0",
             "-frames:v", "1", "-c:v", "h264_nvenc",
             "-f", "null", "-"],
            input=bytes(16 * 16 * 3),
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


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


def _timeline_video_duration(timeline: dict[str, Any]) -> float:
    end_points = [0.0]
    for clip in timeline.get("video_tracks", []):
        end_points.append(float(clip.get("start", 0.0)) + float(clip.get("duration", 0.0)))
    return max(end_points)


def _resolve_video_clip_index(clips: list[dict[str, Any]], clip_index: int) -> int | None:
    if len(clips) == 0:
        return None
    idx = int(clip_index)
    if idx < 0:
        idx = len(clips) + idx
    idx = max(0, min(len(clips) - 1, idx))
    return idx


def _resolve_clip_transition(timeline: dict[str, Any], clip: dict[str, Any], index: int, total: int, entering: bool) -> dict[str, Any] | None:
    key = "transition_in" if entering else "transition_out"
    direct = clip.get(key)
    if isinstance(direct, dict):
        return direct
    global_t = timeline.get("global_transition")
    if not isinstance(global_t, dict):
        return None
    if entering and index == 0 and not bool(global_t.get("apply_to_first", False)):
        return None
    if (not entering) and index == total - 1 and not bool(global_t.get("apply_to_last", False)):
        return None
    return {
        "type": str(global_t.get("type", "crossfade")),
        "duration": _safe_float(global_t.get("duration", 0.35), 0.35, min_value=0.0),
        "easing": str(global_t.get("easing", "ease_in_out")),
        "softness": _safe_float(global_t.get("softness", 0.5), 0.5, min_value=0.0, max_value=1.0),
        "custom_curve": str(global_t.get("custom_curve", "")),
    }


TRANSITION_TYPES = [
    "none",
    "crossfade",
    "fade_black",
    "slide_left",
    "slide_right",
    "slide_up",
    "slide_down",
    "zoom_in",
    "zoom_out",
    "wipe_left",
    "wipe_right",
    "push_left",
    "push_right",
    "custom",
]

TRANSITION_EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out"]

FX_TYPES = [
    "none",
    "shake",
    "zoom_pulse",
    "glitch",
    "strobe",
    "drift",
    "spin_wiggle",
    "custom",
]

FX_PRESETS = [
    "cyber_glitch",
    "impact_shake",
    "dream_pulse",
    "energy_spin",
]

GPU_SHADER_CODECS = ["libx264", "h264_nvenc", "hevc_nvenc"]
GPU_SHADER_BACKENDS = ["libplacebo_vulkan", "libplacebo_auto", "cuda"]


def _get_motion_object(item, prop: str):
    if prop == "position":
        return item.position.enable_motion()
    if prop == "scale":
        return item.scale.enable_motion()
    if prop == "rotation":
        return item.rotation.enable_motion()
    if prop == "opacity":
        return item.opacity.enable_motion()
    raise ValueError(f"Unsupported motion prop: {prop}")


def _safe_times(times, eps: float = 1e-4) -> list[float]:
    out: list[float] = []
    last = None
    for t in times:
        tt = float(t)
        if last is not None and tt <= last:
            tt = last + eps
        out.append(tt)
        last = tt
    return out


def _extend_motion_safe(item, prop: str, keyframes, values, easings=None, clear: bool = False, eps: float = 1e-4):
    motion = _get_motion_object(item, prop)
    if clear:
        motion.clear()
    times = _safe_times([float(t) for t in keyframes], eps=eps)

    existing = set()
    try:
        for k in getattr(motion, "keyframes", []):
            existing.add(round(float(k), 6))
    except Exception:
        pass

    adjusted = []
    for t in times:
        tt = float(t)
        while round(tt, 6) in existing:
            tt += eps
        adjusted.append(tt)
        existing.add(round(tt, 6))

    if easings is not None:
        return motion.extend(keyframes=adjusted, values=values, easings=easings)
    return motion.extend(keyframes=adjusted, values=values)


def _lerp(a, b, x: float):
    return a + (b - a) * x


def _lerp_tuple(a, b, x: float):
    return tuple(_lerp(float(aa), float(bb), x) for aa, bb in zip(a, b))


def _ease_progress(kind: str, x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    k = str(kind or "linear").strip().lower()
    if k == "ease_in":
        return x * x
    if k == "ease_out":
        return 1.0 - (1.0 - x) * (1.0 - x)
    if k == "ease_in_out":
        if x < 0.5:
            return 2.0 * x * x
        return 1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    return x


def _interp_piecewise(times: list[float], values: list[Any], t: float, easings: list[str] | None = None):
    if len(times) == 0 or len(values) == 0:
        return None
    if len(times) == 1 or len(values) == 1:
        return values[0]
    if t <= float(times[0]):
        return values[0]
    if t >= float(times[-1]):
        return values[-1]
    for i in range(len(times) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        if t0 <= t <= t1:
            p = 0.0 if t1 <= t0 else (float(t) - t0) / (t1 - t0)
            if easings and i < len(easings):
                p = _ease_progress(easings[i], p)
            a = values[i]
            b = values[i + 1]
            if isinstance(a, tuple):
                return _lerp_tuple(a, b, p)
            return float(_lerp(float(a), float(b), p))
    return values[-1]


def _sample_motion_property(clip: dict[str, Any], prop: str, t: float, default_value, width: int, height: int):
    motion = clip.get("motion") if isinstance(clip.get("motion"), dict) else None
    if not motion:
        return default_value
    keyframe_times = motion.get("keyframe_times")
    if not isinstance(keyframe_times, list) or len(keyframe_times) < 2:
        return default_value
    norm_times = [max(0.0, min(1.0, float(x))) for x in keyframe_times]
    clip_duration = max(1e-4, float(clip.get("duration", 0.01)))
    abs_times = [clip_duration * nt for nt in norm_times]
    easings = [str(x) for x in motion.get("easings", [])] if isinstance(motion.get("easings"), list) else None

    if prop == "position":
        px = motion.get("position_x", []) if isinstance(motion.get("position_x"), list) else []
        py = motion.get("position_y", []) if isinstance(motion.get("position_y"), list) else []
        if not px and not py:
            return default_value
        vals = []
        base_x = clip.get("position_x", 0.5)
        base_y = clip.get("position_y", 0.5)
        for i in range(len(abs_times)):
            raw_x = px[i] if i < len(px) else base_x
            raw_y = py[i] if i < len(py) else base_y
            vals.append(_position_to_pixels(raw_x, raw_y, width, height))
        return _interp_piecewise(abs_times, vals, t, easings) or default_value

    source = motion.get(prop, []) if isinstance(motion.get(prop), list) else []
    if prop in {"scale_x", "scale_y"}:
        return default_value
    if prop == "scale":
        sx = motion.get("scale_x", []) if isinstance(motion.get("scale_x"), list) else []
        sy = motion.get("scale_y", []) if isinstance(motion.get("scale_y"), list) else []
        if not sx and not sy:
            return default_value
        vals = []
        base_x = clip.get("scale_x", default_value[0])
        base_y = clip.get("scale_y", default_value[1])
        for i in range(len(abs_times)):
            x = _safe_float(sx[i], base_x, min_value=0.01, max_value=20.0) if i < len(sx) else _safe_float(base_x, default_value[0], min_value=0.01, max_value=20.0)
            y = _safe_float(sy[i], base_y, min_value=0.01, max_value=20.0) if i < len(sy) else _safe_float(base_y, default_value[1], min_value=0.01, max_value=20.0)
            vals.append((x, y))
        return _interp_piecewise(abs_times, vals, t, easings) or default_value
    if prop == "rotation":
        if not source:
            source = motion.get("rotation", []) if isinstance(motion.get("rotation"), list) else []
        if not source:
            return default_value
        vals = [_safe_float(source[i], clip.get("rotation", default_value), min_value=-360.0, max_value=360.0) if i < len(source) else default_value for i in range(len(abs_times))]
        return _interp_piecewise(abs_times, vals, t, easings) or default_value
    if prop == "opacity":
        source = motion.get("opacity", []) if isinstance(motion.get("opacity"), list) else []
        if not source:
            return default_value
        vals = [_safe_float(source[i], clip.get("opacity", default_value), min_value=0.0, max_value=1.0) if i < len(source) else default_value for i in range(len(abs_times))]
        return _interp_piecewise(abs_times, vals, t, easings) or default_value
    return default_value


def _effect_weight(effect: dict[str, Any], t: float, clip_duration: float) -> float:
    start_norm = _safe_float(effect.get("start_norm", 0.0), 0.0, min_value=0.0, max_value=1.0)
    end_norm = _safe_float(effect.get("end_norm", 1.0), 1.0, min_value=0.0, max_value=1.0)
    t0 = float(clip_duration) * start_norm
    t1 = float(clip_duration) * max(start_norm + 1e-4, end_norm)
    if t < t0 or t > t1:
        return 0.0
    p = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    return np.sin(np.pi * max(0.0, min(1.0, p)))


def _noise1(seed: float, x: float) -> float:
    return float(np.sin(x * 12.9898 + seed * 78.233) * np.cos(x * 4.123 + seed * 3.17))


def _sample_effect_contribution(effect: dict[str, Any], t: float, clip_duration: float, width: int, height: int, clip_index: int):
    kind = str(effect.get("type", "none")).strip().lower()
    if kind == "none":
        return (0.0, 0.0), (1.0, 1.0), 0.0, 1.0
    w = _effect_weight(effect, t, clip_duration)
    if w <= 0.0:
        return (0.0, 0.0), (1.0, 1.0), 0.0, 1.0

    intensity = _safe_float(effect.get("intensity", 0.5), 0.5, min_value=0.0, max_value=2.0)
    speed = _safe_float(effect.get("speed", 1.0), 1.0, min_value=0.05, max_value=20.0)
    seed = int(effect.get("seed", 0)) + int(clip_index)
    phase = float(t) * speed

    if kind == "shake":
        amp = min(width, height) * 0.006 * intensity * w
        dx = amp * _noise1(seed + 1.0, phase * 7.0)
        dy = amp * _noise1(seed + 2.0, phase * 9.0 + 3.0)
        return (dx, dy), (1.0, 1.0), 0.0, 1.0

    if kind == "zoom_pulse":
        amp = 0.03 * intensity * w
        k = 1.0 + amp * np.sin(phase * np.pi * 2.0)
        return (0.0, 0.0), (float(k), float(k)), 0.0, 1.0

    if kind == "strobe":
        hz = max(2.0, speed * 8.0)
        gate = 1.0 if int(np.floor((t + seed * 0.001) * hz)) % 2 == 0 else max(0.0, 1.0 - 0.75 * intensity)
        alpha_mult = float(_lerp(1.0, gate, w))
        return (0.0, 0.0), (1.0, 1.0), 0.0, alpha_mult

    if kind == "drift":
        dx = width * 0.04 * intensity * w * np.sin(phase * np.pi)
        dy = -height * 0.02 * intensity * w * np.sin(phase * np.pi)
        return (float(dx), float(dy)), (1.0, 1.0), 0.0, 1.0

    if kind == "spin_wiggle":
        deg = 8.0 * intensity * w * np.sin(phase * np.pi * 2.0)
        return (0.0, 0.0), (1.0, 1.0), float(deg), 1.0

    if kind in {"glitch", "custom"}:
        amp = min(width, height) * 0.008 * intensity * w
        dx = amp * _noise1(seed + 3.0, phase * 11.0)
        dy = amp * _noise1(seed + 4.0, phase * 13.0 + 1.2)
        rot = 3.5 * intensity * w * _noise1(seed + 5.0, phase * 5.0)
        alpha_mult = max(0.0, 1.0 - 0.45 * intensity * w * (0.5 + 0.5 * _noise1(seed + 6.0, phase * 17.0)))
        if kind == "custom":
            parsed = _parse_custom_curve(str(effect.get("custom_curve", "")))
            if parsed:
                ts, vs = parsed
                start_norm = _safe_float(effect.get("start_norm", 0.0), 0.0, min_value=0.0, max_value=1.0)
                end_norm = _safe_float(effect.get("end_norm", 1.0), 1.0, min_value=0.0, max_value=1.0)
                t0 = float(clip_duration) * start_norm
                t1 = float(clip_duration) * max(start_norm + 1e-4, end_norm)
                local_p = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                vv = _interp_piecewise(ts, vs, local_p, None)
                if vv is not None:
                    dx = (float(vv) - 0.5) * 2.0 * amp
                    dy = (0.5 - float(vv)) * 2.0 * amp
        return (float(dx), float(dy)), (1.0, 1.0), float(rot), float(alpha_mult)

    return (0.0, 0.0), (1.0, 1.0), 0.0, 1.0


def _collect_advanced_sample_times(clip: dict[str, Any], clip_duration: float, fps: float, transitions: list[dict[str, Any]]):
    base_rate = min(max(float(fps) * 2.0, 24.0), 120.0)
    steps = max(12, int(np.ceil(float(clip_duration) * base_rate)) + 1)
    times = set(float(x) for x in np.linspace(0.0, float(clip_duration), steps))
    times.update([0.0, float(clip_duration)])

    motion = clip.get("motion") if isinstance(clip.get("motion"), dict) else None
    if motion and isinstance(motion.get("keyframe_times"), list):
        for nt in motion.get("keyframe_times", []):
            times.add(max(0.0, min(float(clip_duration), float(clip_duration) * max(0.0, min(1.0, float(nt))))))

    for tr in transitions:
        if not tr:
            continue
        dur = min(float(clip_duration), _safe_float(tr.get("duration", 0.0), 0.0, min_value=0.0))
        if dur <= 0:
            continue
        times.update([0.0, dur, max(0.0, float(clip_duration) - dur), float(clip_duration)])

    effects = clip.get("advanced_fx_layers", [])
    if isinstance(effects, list):
        for fx in effects:
            if not isinstance(fx, dict):
                continue
            s = _safe_float(fx.get("start_norm", 0.0), 0.0, min_value=0.0, max_value=1.0)
            e = _safe_float(fx.get("end_norm", 1.0), 1.0, min_value=0.0, max_value=1.0)
            t0 = float(clip_duration) * s
            t1 = float(clip_duration) * max(s + 1e-4, e)
            tm = t0 + (t1 - t0) * 0.5
            times.update([t0, tm, t1])

    return sorted(times)


def _compute_transition_state(transition: dict[str, Any] | None, t: float, clip_duration: float, base_position: tuple[float, float], base_scale: tuple[float, float], base_opacity: float, width: int, height: int, entering: bool):
    pos = base_position
    scale = base_scale
    opacity = base_opacity
    if not transition or str(transition.get("type", "none")) == "none":
        return pos, scale, opacity

    kind = str(transition.get("type", "none"))
    dur = min(float(clip_duration), _safe_float(transition.get("duration", 0.0), 0.0, min_value=0.0))
    if dur <= 0:
        return pos, scale, opacity
    easing = str(transition.get("easing", "ease_in_out"))
    softness = _safe_float(transition.get("softness", 0.5), 0.5, min_value=0.0, max_value=1.0)
    custom_curve = str(transition.get("custom_curve", ""))

    if entering:
        if t > dur:
            return pos, scale, opacity
        p = 0.0 if dur <= 0 else t / dur
    else:
        t0 = max(0.0, float(clip_duration) - dur)
        if t < t0:
            return pos, scale, opacity
        p = 1.0 if dur <= 0 else (t - t0) / dur

    if kind == "custom":
        parsed = _parse_custom_curve(custom_curve)
        if parsed:
            ts, vs = parsed
            v = _interp_piecewise(ts, vs, p, None)
            if v is not None:
                opacity = float(base_opacity) * float(v)
            return pos, scale, opacity
        kind = "crossfade"

    if easing == "linear":
        op_progress = p
    elif easing == "ease_in":
        op_progress = _ease_progress("ease_in", p)
    elif easing == "ease_out":
        op_progress = _ease_progress("ease_out", p)
    else:
        mid_p = 0.5 + softness * 0.15
        if p < mid_p:
            op_progress = 0.5 * _ease_progress("ease_in", p / max(1e-4, mid_p))
        else:
            op_progress = 0.5 + 0.5 * _ease_progress("ease_out", (p - mid_p) / max(1e-4, 1.0 - mid_p))

    opacity = float(_lerp(0.0, base_opacity, op_progress) if entering else _lerp(base_opacity, 0.0, op_progress))

    if kind in {"slide_left", "wipe_left", "push_left", "slide_right", "wipe_right", "push_right", "slide_up", "slide_down"}:
        p0, p1 = _transition_motion_values(kind, entering, base_position, base_scale, width, height)
        pos = _lerp_tuple(p0, p1, _ease_progress(easing, p))
    elif kind in {"zoom_in", "zoom_out"}:
        s0, s1 = _transition_motion_values(kind, entering, base_position, base_scale, width, height)
        scale = _lerp_tuple(s0, s1, _ease_progress(easing, p))
    return pos, scale, opacity


def _apply_advanced_clip_animation(item, timeline: dict[str, Any], clip: dict[str, Any], clip_duration: float, base_position: tuple[float, float], base_scale: tuple[float, float], base_rotation: float, base_opacity: float, width: int, height: int, fps: float, clip_index: int, total_clips: int):
    transition_in = _resolve_clip_transition(timeline, clip, clip_index, total_clips, entering=True)
    transition_out = _resolve_clip_transition(timeline, clip, clip_index, total_clips, entering=False)
    times = _collect_advanced_sample_times(clip, clip_duration, fps, [transition_in, transition_out])

    pos_vals = []
    scale_vals = []
    rot_vals = []
    op_vals = []

    fx_layers = clip.get("advanced_fx_layers", [])
    if not isinstance(fx_layers, list):
        fx_layers = []

    for t in times:
        pos = _sample_motion_property(clip, "position", t, base_position, width, height)
        scl = _sample_motion_property(clip, "scale", t, base_scale, width, height)
        rot = _sample_motion_property(clip, "rotation", t, base_rotation, width, height)
        op = _sample_motion_property(clip, "opacity", t, base_opacity, width, height)

        pos, scl, op = _compute_transition_state(transition_in, t, clip_duration, pos, scl, op, width, height, entering=True)
        pos, scl, op = _compute_transition_state(transition_out, t, clip_duration, pos, scl, op, width, height, entering=False)

        if transition_in is None and _safe_float(clip.get("fade_in", 0.0), 0.0, min_value=0.0) > 0.0:
            fin = min(float(clip_duration), _safe_float(clip.get("fade_in", 0.0), 0.0, min_value=0.0))
            if t <= fin:
                op *= (t / max(1e-4, fin))
        if transition_out is None and _safe_float(clip.get("fade_out", 0.0), 0.0, min_value=0.0) > 0.0:
            fout = min(float(clip_duration), _safe_float(clip.get("fade_out", 0.0), 0.0, min_value=0.0))
            t0 = max(0.0, float(clip_duration) - fout)
            if t >= t0:
                op *= max(0.0, 1.0 - ((t - t0) / max(1e-4, fout)))

        total_dx = 0.0
        total_dy = 0.0
        total_rot = 0.0
        total_op_mult = 1.0
        scale_mult_x = 1.0
        scale_mult_y = 1.0
        for fx in fx_layers:
            if not isinstance(fx, dict):
                continue
            (dx, dy), (mx, my), drot, op_mult = _sample_effect_contribution(fx, t, clip_duration, width, height, clip_index)
            total_dx += dx
            total_dy += dy
            scale_mult_x *= mx
            scale_mult_y *= my
            total_rot += drot
            total_op_mult *= op_mult

        pos = (float(pos[0] + total_dx), float(pos[1] + total_dy))
        scl = (float(scl[0] * scale_mult_x), float(scl[1] * scale_mult_y))
        rot = float(rot + total_rot)
        op = float(max(0.0, min(1.0, op * total_op_mult)))

        pos_vals.append(pos)
        scale_vals.append(scl)
        rot_vals.append(rot)
        op_vals.append(op)

    _extend_motion_safe(item, "position", times, pos_vals, clear=True)
    _extend_motion_safe(item, "scale", times, scale_vals, clear=True)
    _extend_motion_safe(item, "rotation", times, rot_vals, clear=True)
    _extend_motion_safe(item, "opacity", times, op_vals, clear=True)



def _escape_ffmpeg_filter_path(path: str) -> str:
    p = str(Path(path).resolve()).replace("\\", "/")
    p = p.replace(":", "\\:").replace("'", "\\'")
    return p


def _resolve_shader_file(shader_path: str, shader_code: str, prefix: str = "movis_shader") -> str:
    code = str(shader_code or "").strip()
    if code:
        temp_dir = Path(folder_paths.get_temp_directory()) / "vhs_movis_shaders"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out = temp_dir / f"{prefix}_{_now_stamp()}.glsl"
        out.write_text(code, encoding="utf-8")
        return str(out.resolve())
    path = str(shader_path or "").strip()
    if not path:
        raise ValueError("未提供 shader：请设置 shader_path 或 shader_code")
    return resolve_media_path(path, must_exist=True)


def _check_vulkan_available() -> bool:
    """Check if Vulkan is usable on this system for ffmpeg libplacebo."""
    if sys.platform == "win32":
        return True
    try:
        res = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=10,
        )
        out = (res.stdout + res.stderr).lower()
        if "devicename" in out and "llvmpipe" not in out:
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ---------------------------------------------------------------------------
# mpv/libplacebo shader → standard GLSL 330 converter
# ---------------------------------------------------------------------------

def _parse_mpv_shader_to_fragment(shader_text: str) -> str:
    """
    Best-effort conversion of an mpv / libplacebo HOOK shader to a
    standard GLSL 330 core fragment shader that can be run via moderngl.

    If the file is already standard GLSL (contains ``void main``), it is
    returned as-is.
    """
    # Already standard GLSL?
    if "//!HOOK" not in shader_text and "//!BIND" not in shader_text:
        return shader_text

    # Strip mpv directives, keep the rest
    body_lines: list[str] = []
    for line in shader_text.split("\n"):
        if line.strip().startswith("//!"):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)

    # Replace mpv builtins with standard GLSL equivalents
    body = body.replace("HOOKED_tex(HOOKED_pos)", "texture(u_tex, v_uv)")
    body = body.replace("HOOKED_pos", "v_uv")
    body = body.replace("HOOKED_size", "u_tex_size")
    body = body.replace("HOOKED_pt", "u_tex_pt")
    # HOOKED_texOff(0) and similar zero-offset forms
    body = re.sub(
        r"HOOKED_texOff\(\s*(?:0|vec2\s*\(\s*0(?:\.0)?\s*(?:,\s*0(?:\.0)?\s*)?\))\s*\)",
        "texture(u_tex, v_uv)",
        body,
    )
    # General HOOKED_texOff(expr) → texture(u_tex, v_uv + (expr) * u_tex_pt)
    body = re.sub(
        r"HOOKED_texOff\(([^)]+)\)",
        r"texture(u_tex, v_uv + (\1) * u_tex_pt)",
        body,
    )
    # HOOKED_tex(expr) → texture(u_tex, expr)
    body = re.sub(
        r"HOOKED_tex\(([^)]+)\)",
        r"texture(u_tex, \1)",
        body,
    )

    has_hook_fn = bool(re.search(r"vec4\s+hook\s*\(", body))

    header = (
        "#version 330 core\n"
        "precision mediump float;\n"
        "uniform sampler2D u_tex;\n"
        "uniform vec2 u_tex_size;\n"
        "uniform vec2 u_tex_pt;\n"
        "in vec2 v_uv;\n"
        "out vec4 fragColor;\n\n"
    )

    if has_hook_fn:
        return header + body + "\nvoid main() { fragColor = hook(); }\n"
    else:
        # Assume an already-complete main or inline code
        return header + body + "\n"


# ---------------------------------------------------------------------------
# OpenGL (moderngl / EGL) frame-by-frame shader renderer
# ---------------------------------------------------------------------------

def _apply_shader_opengl(
    input_video_path: str,
    shader_file_path: str,
    output_path: str,
    codec: str,
    keep_audio: bool,
) -> str:
    """
    Process every frame of *input_video_path* through a GLSL shader using
    **moderngl** with an EGL (headless) or native OpenGL context – the same
    mechanism DepthFlow / ShaderFlow uses.

    Falls back gracefully if moderngl is not installed.
    """
    try:
        import moderngl
    except ImportError:
        raise RuntimeError(
            "OpenGL shader回退需要 moderngl 包。\n"
            "请在 ComfyUI 虚拟环境中安装: pip install moderngl\n"
            "（DepthFlow之所以能渲染shader就是因为用了moderngl+EGL）"
        )
    import cv2 as _cv2

    # --- Read & convert shader ---
    shader_text = Path(shader_file_path).read_text(encoding="utf-8")
    frag_src = _parse_mpv_shader_to_fragment(shader_text)

    vert_src = (
        "#version 330 core\n"
        "in vec2 in_pos;\n"
        "in vec2 in_uv;\n"
        "out vec2 v_uv;\n"
        "void main() {\n"
        "    gl_Position = vec4(in_pos, 0.0, 1.0);\n"
        "    v_uv = in_uv;\n"
        "}\n"
    )

    # --- Open video ---
    cap = _cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_video_path}")

    width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

    # --- Create GL context ---
    ctx = None
    for backend in ("egl", None):  # try EGL first, then default
        try:
            if backend:
                ctx = moderngl.create_standalone_context(backend=backend)
            else:
                ctx = moderngl.create_standalone_context()
            break
        except Exception:
            continue
    if ctx is None:
        cap.release()
        raise RuntimeError("无法创建 OpenGL 上下文（EGL 和默认后端均失败）")

    try:
        prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    except Exception as e:
        ctx.release()
        cap.release()
        raise RuntimeError(f"GLSL shader编译失败:\n{e}\n\n转换后的fragment shader:\n{frag_src[:500]}")

    # Full-screen quad (pos_x, pos_y, uv_x, uv_y)
    quad = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype="f4")
    vbo = ctx.buffer(quad)
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos", "in_uv")

    tex = ctx.texture((width, height), 3)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    color_att = ctx.texture((width, height), 4)
    fbo = ctx.framebuffer(color_attachments=[color_att])

    # Uniforms (set only if present in the shader)
    if "u_tex" in prog:
        prog["u_tex"].value = 0
    if "u_tex_size" in prog:
        prog["u_tex_size"].value = (float(width), float(height))
    if "u_tex_pt" in prog:
        prog["u_tex_pt"].value = (1.0 / width, 1.0 / height)

    # --- Encode output (temp without audio) ---
    temp_path = output_path + ".tmp_gl.mp4"
    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
    writer = _cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        ctx.release()
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {temp_path}")

    frame_idx = 0
    print(f"[VHS OpenGL] 开始帧处理 ({width}x{height}, {total_frames} frames)…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR→RGB, flip Y for OpenGL
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(np.flip(rgb, axis=0))
        tex.write(rgb.tobytes())

        fbo.use()
        tex.use(location=0)
        vao.render(moderngl.TRIANGLE_STRIP)

        data = fbo.read(components=4)
        out_frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        out_frame = np.flip(out_frame, axis=0)
        bgr = _cv2.cvtColor(out_frame, _cv2.COLOR_RGBA2BGR)
        writer.write(bgr)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[VHS OpenGL] {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    tex.release()
    color_att.release()
    fbo.release()
    vbo.release()
    vao.release()
    ctx.release()
    print(f"[VHS OpenGL] 帧处理完成 ({frame_idx} frames)")

    # --- Mux audio back if needed ---
    if keep_audio:
        _ffmpeg_bin = ffmpeg_path or "ffmpeg"
        mux_cmd = [
            _ffmpeg_bin, "-y", "-v", "error",
            "-i", temp_path,
            "-i", input_video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a?",
            output_path,
        ]
        mux = subprocess.run(mux_cmd, capture_output=True, text=True)
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass
        if mux.returncode != 0:
            raise RuntimeError(f"OpenGL shader完成但音频混流失败:\n{mux.stderr.strip()}")
    else:
        Path(temp_path).replace(output_path)

    return output_path


# ---------------------------------------------------------------------------
# CUDA (PyTorch) shader renderer — uses GPU tensor ops, no OpenGL needed
# ---------------------------------------------------------------------------

def _glsl_to_cuda_shader_fn(shader_file_path: str):
    """
    Parse a GLSL shader and return a PyTorch-compatible function that
    processes a (H, W, 3) float32 CUDA tensor and returns a (H, W, 3)
    float32 CUDA tensor.

    Supports two modes:
      1. **Known shader patterns**: detects common mpv/libplacebo shaders
         (sharpen, blur, color grading, etc.) and maps them to optimized
         PyTorch equivalents.
      2. **Generic passthrough + shader effects**: for shaders we cannot
         parse, applies the shader as best-effort per-pixel math or
         falls back to identity (with a warning).

    This is NOT a full GLSL interpreter — it covers the most common
    shader effects used in video post-processing.
    """
    import torch
    import torch.nn.functional as F

    shader_text = Path(shader_file_path).read_text(encoding="utf-8")
    shader_lower = shader_text.lower()

    # --- Detect shader type by keyword analysis ---

    def _is_sharpen():
        return any(kw in shader_lower for kw in (
            "sharpen", "cas", "unsharp", "adaptive_sharpen",
            "contrast adaptive", "luma_sharpen",
        ))

    def _is_blur():
        return any(kw in shader_lower for kw in (
            "blur", "gaussian", "box_blur", "kawase",
        )) and "unsharp" not in shader_lower

    def _is_color_grade():
        return any(kw in shader_lower for kw in (
            "color", "gamma", "contrast", "brightness", "saturation",
            "vibrance", "tone", "lut", "grade",
        ))

    def _is_vignette():
        return "vignette" in shader_lower

    def _is_grain():
        return any(kw in shader_lower for kw in ("grain", "noise", "film_grain"))

    # --- Build the CUDA function ---

    if _is_sharpen():
        # Extract strength if possible
        import re as _re
        m = _re.search(r'(?:strength|sharp|amount|intensity)\s*[=:]\s*([0-9.]+)', shader_text)
        strength = float(m.group(1)) if m else 0.5
        strength = max(0.1, min(2.0, strength))
        print(f"[VHS CUDA] Detected SHARPEN shader (strength={strength:.2f})")

        def sharpen_fn(frame_tensor):
            # frame_tensor: (H, W, 3) float32 CUDA, range [0, 1]
            t = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            # Unsharp mask: sharp = original + strength * (original - blur)
            blurred = F.avg_pool2d(F.pad(t, (1,1,1,1), mode='reflect'), 3, stride=1)
            sharpened = t + strength * (t - blurred)
            sharpened = sharpened.clamp(0.0, 1.0)
            return sharpened.squeeze(0).permute(1, 2, 0)  # (H, W, 3)

        return sharpen_fn

    if _is_blur():
        import re as _re
        m = _re.search(r'(?:radius|sigma|size|kernel)\s*[=:]\s*([0-9.]+)', shader_text)
        radius = int(float(m.group(1))) if m else 3
        radius = max(1, min(15, radius))
        ksize = radius * 2 + 1
        print(f"[VHS CUDA] Detected BLUR shader (kernel={ksize})")

        def blur_fn(frame_tensor):
            t = frame_tensor.permute(2, 0, 1).unsqueeze(0)
            pad = ksize // 2
            blurred = F.avg_pool2d(F.pad(t, (pad,pad,pad,pad), mode='reflect'),
                                   ksize, stride=1)
            return blurred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

        return blur_fn

    if _is_color_grade():
        import re as _re
        gamma = 1.0
        contrast = 1.0
        brightness = 0.0
        saturation = 1.0
        m = _re.search(r'gamma\s*[=:]\s*([0-9.]+)', shader_text)
        if m: gamma = max(0.1, min(3.0, float(m.group(1))))
        m = _re.search(r'contrast\s*[=:]\s*([0-9.]+)', shader_text)
        if m: contrast = max(0.1, min(3.0, float(m.group(1))))
        m = _re.search(r'brightness\s*[=:]\s*([0-9.-]+)', shader_text)
        if m: brightness = max(-1.0, min(1.0, float(m.group(1))))
        m = _re.search(r'saturation\s*[=:]\s*([0-9.]+)', shader_text)
        if m: saturation = max(0.0, min(3.0, float(m.group(1))))
        print(f"[VHS CUDA] Detected COLOR shader (gamma={gamma:.2f}, contrast={contrast:.2f}, "
              f"brightness={brightness:.2f}, saturation={saturation:.2f})")

        def color_fn(frame_tensor):
            t = frame_tensor.clone()
            # Gamma
            if gamma != 1.0:
                t = t.clamp(1e-6, 1.0).pow(1.0 / gamma)
            # Contrast + brightness
            if contrast != 1.0 or brightness != 0.0:
                t = (t - 0.5) * contrast + 0.5 + brightness
            # Saturation
            if saturation != 1.0:
                luma = t[..., 0] * 0.2126 + t[..., 1] * 0.7152 + t[..., 2] * 0.0722
                t = luma.unsqueeze(-1) + saturation * (t - luma.unsqueeze(-1))
            return t.clamp(0.0, 1.0)

        return color_fn

    if _is_vignette():
        print("[VHS CUDA] Detected VIGNETTE shader")

        def vignette_fn(frame_tensor):
            H, W, _ = frame_tensor.shape
            device = frame_tensor.device
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            d = (xx*xx + yy*yy).sqrt()
            vignette = 1.0 - (d * 0.5).clamp(0, 1)
            return (frame_tensor * vignette.unsqueeze(-1)).clamp(0.0, 1.0)

        return vignette_fn

    if _is_grain():
        print("[VHS CUDA] Detected FILM GRAIN shader")

        def grain_fn(frame_tensor):
            noise = torch.randn_like(frame_tensor) * 0.03
            return (frame_tensor + noise).clamp(0.0, 1.0)

        return grain_fn

    # --- Fallback: identity with warning ---
    print(f"[VHS CUDA] ⚠ Unrecognized shader type — applying as identity pass")
    print(f"[VHS CUDA]   (Shader will still be applied via OpenGL fallback if available)")
    return None  # Signal caller to try other methods


def _apply_shader_cuda(
    input_video_path: str,
    shader_file_path: str,
    output_path: str,
    codec: str,
    keep_audio: bool,
) -> str:
    """
    Apply a shader effect to video using CUDA/PyTorch tensor operations.
    Reads frames via cv2, processes on GPU with PyTorch, encodes output
    via ffmpeg pipe (NVENC when available).

    This is blazing fast because:
    - Frame decode: CPU (cv2)
    - Frame processing: GPU (PyTorch CUDA tensors)
    - Frame encode: GPU (NVENC via ffmpeg) or CPU (libx264)
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    import cv2 as _cv2

    shader_fn = _glsl_to_cuda_shader_fn(shader_file_path)
    if shader_fn is None:
        raise RuntimeError("Shader not recognized for CUDA mode")

    device = torch.device("cuda")

    # Open input
    cap = _cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

    _ffmpeg_bin = ffmpeg_path or "ffmpeg"

    # Try NVENC first, then fall back to libx264
    nvenc_map = {"libx264": "h264_nvenc", "h264_nvenc": "h264_nvenc", "hevc_nvenc": "hevc_nvenc"}
    enc = nvenc_map.get(codec, "h264_nvenc")
    nvenc_ok = True

    # Build ffmpeg pipe command for encoding
    temp_path = output_path + ".tmp_cuda.mp4"
    encode_cmd = [
        _ffmpeg_bin, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", enc,
        "-pix_fmt", "yuv420p",
        "-an",
        temp_path,
    ]
    try:
        proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        nvenc_ok = False

    if nvenc_ok:
        # Quick test: write a dummy frame
        test_frame = b'\x00' * (width * height * 3)
        try:
            proc.stdin.write(test_frame)
        except Exception:
            nvenc_ok = False
            proc.kill()

    if not nvenc_ok:
        # Fall back to libx264
        enc = "libx264"
        encode_cmd = [
            _ffmpeg_bin, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", enc,
            "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            temp_path,
        ]
        proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # If we had to restart, also reopen video (we already wrote 1 junk frame)
    if not nvenc_ok:
        cap.release()
        cap = _cv2.VideoCapture(input_video_path)
    else:
        # We already wrote a test frame — restart cleanly
        proc.stdin.close()
        proc.wait()
        cap.release()
        cap = _cv2.VideoCapture(input_video_path)
        proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"[VHS CUDA] Processing {total_frames} frames @ {width}x{height} (encoder: {enc})")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR→RGB, to float32 [0,1], move to CUDA
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).float().div_(255.0).to(device)

            # Apply shader
            out_t = shader_fn(t)

            # Back to uint8 numpy
            out_np = (out_t.clamp(0.0, 1.0).mul_(255.0).byte().cpu().numpy())

            # Write to ffmpeg pipe
            proc.stdin.write(out_np.tobytes())

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"[VHS CUDA] {frame_idx}/{total_frames}")

    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()

    print(f"[VHS CUDA] Processed {frame_idx} frames")

    # Mux audio
    if keep_audio:
        mux_cmd = [
            _ffmpeg_bin, "-y", "-v", "error",
            "-i", temp_path,
            "-i", input_video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a?",
            output_path,
        ]
        r = subprocess.run(mux_cmd, capture_output=True, text=True)
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass
        if r.returncode != 0:
            raise RuntimeError(f"Audio mux failed: {r.stderr.strip()}")
    else:
        Path(temp_path).replace(output_path)

    return output_path


# ---------------------------------------------------------------------------
# Normalize clip shader entry (used by MovisSetClipShader)
# ---------------------------------------------------------------------------

def _normalize_clip_shader_entry(entry: dict) -> dict | None:
    """Validate and normalize a clip shader configuration dict."""
    if not isinstance(entry, dict):
        return None
    if not entry.get("enabled", True):
        return None
    sp = str(entry.get("shader_path", "") or "").strip()
    sc = str(entry.get("shader_code", "") or "").strip()
    if not sp and not sc:
        return None
    return {
        "shader_path": sp,
        "shader_code": sc,
        "backend": str(entry.get("backend", "libplacebo_vulkan")).strip(),
        "codec": str(entry.get("codec", "libx264")).strip(),
        "keep_audio": bool(entry.get("keep_audio", True)),
        "enabled": True,
        "output_file_prefix": str(entry.get("output_file_prefix", "movis_clip_shader_")).strip(),
    }


# ---------------------------------------------------------------------------
# Main entry: apply GLSL shader to video
# ---------------------------------------------------------------------------

def _apply_gpu_shader_ffmpeg(
    input_video_path: str,
    shader_file_path: str,
    output_file_prefix: str,
    backend: str,
    codec: str,
    keep_audio: bool,
) -> str:
    """
    Apply a GLSL shader to a video.

    Strategy (stops at first success):
      1. ffmpeg libplacebo (normal Vulkan)
      2. ffmpeg libplacebo + ``-init_hw_device vulkan``
      3. ffmpeg libplacebo + ``allow_sw=1`` (newer ffmpeg)
      4. ffmpeg libplacebo forced through lavapipe ICD
      5. CUDA/PyTorch frame-by-frame (fast on NVIDIA GPU without OpenGL)
      6. Frame-by-frame OpenGL via moderngl/EGL (same as DepthFlow)

    If backend == "cuda", skips libplacebo strategies and goes straight
    to the CUDA path.
    """
    ffmpeg_bin = ffmpeg_path or "ffmpeg"
    out_name = f"{output_file_prefix}{_now_stamp()}.mp4"
    out_path = str((Path(folder_paths.get_output_directory()) / out_name).resolve())

    c = str(codec or "libx264").strip().lower()
    if c not in GPU_SHADER_CODECS:
        c = "libx264"

    be = str(backend or "libplacebo_vulkan").strip().lower()
    errors: list[str] = []

    # If user explicitly selected CUDA, try it first
    if be == "cuda":
        print("[VHS] Backend=CUDA — trying PyTorch CUDA shader processing…")
        try:
            result = _apply_shader_cuda(
                input_video_path, shader_file_path, out_path, c, keep_audio,
            )
            print("[VHS] CUDA shader渲染成功 ✓")
            return result
        except Exception as cuda_err:
            errors.append(f"CUDA(explicit): {cuda_err}")
            print(f"[VHS] CUDA explicit failed: {cuda_err}")
            # Fall through to libplacebo strategies

    shader_escaped = _escape_ffmpeg_filter_path(shader_file_path)
    vf = f"libplacebo=custom_shader_path='{shader_escaped}'"
    audio_args = (["-map", "0:v:0", "-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"])

    def _run(cmd, *, env=None):
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    # ---- Strategy 1: normal libplacebo ----
    cmd1 = [ffmpeg_bin, "-y", "-v", "error", "-i", input_video_path,
            "-vf", vf, "-c:v", c, "-pix_fmt", "yuv420p"] + audio_args + [out_path]
    r = _run(cmd1)
    if r.returncode == 0:
        return out_path
    errors.append(f"libplacebo: {r.stderr.strip()[:200]}")

    stderr_lower = r.stderr.lower()
    is_vulkan_issue = any(kw in stderr_lower for kw in (
        "vulkan", "no suitable device", "failed creating",
        "failed initializing", "vk_icd", "libplacebo",
    ))
    if not is_vulkan_issue:
        raise RuntimeError(
            f"GPU Shader 渲染失败（非Vulkan问题）。\n"
            f"stderr:\n{r.stderr.strip()}"
        )

    print("[VHS] libplacebo/Vulkan 常规模式失败，尝试备选方案…")

    # ---- Strategy 2: explicit hw device init ----
    cmd2 = [ffmpeg_bin, "-y", "-v", "error",
            "-init_hw_device", "vulkan=vk:0",
            "-i", input_video_path,
            "-vf", vf, "-c:v", c, "-pix_fmt", "yuv420p"] + audio_args + [out_path]
    r2 = _run(cmd2)
    if r2.returncode == 0:
        print("[VHS] libplacebo(hw_device init)成功")
        return out_path
    errors.append(f"hw_device: {r2.stderr.strip()[:200]}")

    # ---- Strategy 3: allow_sw=1 (newer ffmpeg ≥6.x) ----
    vf_sw = f"libplacebo=custom_shader_path='{shader_escaped}':allow_sw=1"
    cmd3 = [ffmpeg_bin, "-y", "-v", "error", "-i", input_video_path,
            "-vf", vf_sw, "-c:v", c, "-pix_fmt", "yuv420p"] + audio_args + [out_path]
    r3 = _run(cmd3)
    if r3.returncode == 0:
        print("[VHS] libplacebo(software Vulkan)成功")
        return out_path
    errors.append(f"allow_sw: {r3.stderr.strip()[:200]}")

    # ---- Strategy 4: force lavapipe ICD ----
    if sys.platform != "win32":
        icd_candidates = [
            "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "/usr/share/vulkan/icd.d/lvp_icd.i686.json",
            "/etc/vulkan/icd.d/lvp_icd.x86_64.json",
            "/usr/share/vulkan/icd.d/lvp_icd.json",
        ]
        for icd in icd_candidates:
            if Path(icd).is_file():
                env4 = os.environ.copy()
                env4["VK_ICD_FILENAMES"] = icd
                env4["VK_DRIVER_FILES"] = icd
                r4 = _run(cmd1, env=env4)
                if r4.returncode == 0:
                    print(f"[VHS] libplacebo(lavapipe ICD={icd})成功")
                    return out_path
                errors.append(f"lavapipe({icd}): {r4.stderr.strip()[:150]}")
                break

    # ---- Strategy 5: CUDA/PyTorch (fast on NVIDIA without OpenGL) ----
    if be != "cuda":  # Skip if already tried above
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                print("[VHS] 尝试 CUDA/PyTorch shader处理…")
                result = _apply_shader_cuda(
                    input_video_path, shader_file_path, out_path, c, keep_audio,
                )
                print("[VHS] CUDA shader渲染成功 ✓")
                return result
        except Exception as cuda_err:
            errors.append(f"CUDA: {cuda_err}")

    # ---- Strategy 6: OpenGL frame-by-frame via moderngl (like DepthFlow) ----
    print("[VHS] 尝试 OpenGL(EGL) 逐帧渲染…")
    try:
        result = _apply_shader_opengl(
            input_video_path, shader_file_path, out_path, c, keep_audio,
        )
        print("[VHS] OpenGL shader渲染成功 ✓")
        return result
    except Exception as gl_err:
        errors.append(f"OpenGL: {gl_err}")

    # ---- All strategies exhausted ----
    detail = "\n".join(f"  {i+1}) {e}" for i, e in enumerate(errors))
    raise RuntimeError(
        f"GPU Shader 渲染失败，所有方案均不可用:\n{detail}\n\n"
        f"解决方法:\n"
        f"  - 安装支持 Vulkan 的 ffmpeg\n"
        f"  - 或设置 backend=cuda 用PyTorch CUDA处理\n"
        f"  - 或: pip install moderngl (启用OpenGL回退)\n"
        f"  - Linux headless请确保安装 libegl1-mesa-dev\n"
    )


def _make_transition(kind: str, duration: float, easing: str = "ease_in_out", softness: float = 0.5, custom_curve: str = "") -> dict[str, Any] | None:
    k = str(kind or "none").strip().lower()
    d = _safe_float(duration, 0.0, min_value=0.0)
    if k == "none" or d <= 0:
        return None
    if k not in TRANSITION_TYPES:
        k = "crossfade"
    e = str(easing or "ease_in_out").strip().lower()
    if e not in TRANSITION_EASINGS:
        e = "ease_in_out"
    return {
        "type": k,
        "duration": d,
        "easing": e,
        "softness": _safe_float(softness, 0.5, min_value=0.0, max_value=1.0),
        "custom_curve": str(custom_curve or "").strip(),
    }


def _parse_custom_curve(curve: str) -> tuple[list[float], list[float]] | None:
    # format: "0:0,0.4:0.2,1:1"
    if not curve:
        return None
    pairs: list[tuple[float, float]] = []
    for seg in str(curve).replace(";", ",").split(","):
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue
        a, b = seg.split(":", 1)
        try:
            t = max(0.0, min(1.0, float(a.strip())))
            v = max(0.0, min(1.0, float(b.strip())))
        except Exception:
            continue
        pairs.append((t, v))
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda x: x[0])
    return [x[0] for x in pairs], [x[1] for x in pairs]


def _make_fx(kind: str, start_norm: float, end_norm: float, intensity: float, speed: float, seed: int = 0, custom_curve: str = "") -> dict[str, Any] | None:
    k = str(kind or "none").strip().lower()
    if k == "none":
        return None
    if k not in FX_TYPES:
        k = "glitch"
    s = _safe_float(start_norm, 0.0, min_value=0.0, max_value=1.0)
    e = _safe_float(end_norm, 1.0, min_value=0.0, max_value=1.0)
    if e <= s:
        e = min(1.0, s + 0.01)
    return {
        "type": k,
        "start_norm": s,
        "end_norm": e,
        "intensity": _safe_float(intensity, 0.5, min_value=0.0, max_value=2.0),
        "speed": _safe_float(speed, 1.0, min_value=0.05, max_value=20.0),
        "seed": int(seed),
        "custom_curve": str(custom_curve or "").strip(),
    }


def _iter_clip_effects(clip: dict[str, Any]) -> list[dict[str, Any]]:
    effects = clip.get("effects")
    if not isinstance(effects, list):
        return []
    out: list[dict[str, Any]] = []
    for fx in effects:
        if isinstance(fx, dict) and fx.get("type"):
            out.append(fx)
    return out


def _apply_clip_fx(item, effect: dict[str, Any], clip_duration: float, base_position: tuple[float, float], base_scale: tuple[float, float], base_rotation: float, base_opacity: float, width: int, height: int, clip_index: int):
    kind = str(effect.get("type", "none")).strip().lower()
    if kind == "none":
        return
    start_norm = _safe_float(effect.get("start_norm", 0.0), 0.0, min_value=0.0, max_value=1.0)
    end_norm = _safe_float(effect.get("end_norm", 1.0), 1.0, min_value=0.0, max_value=1.0)
    t0 = float(clip_duration) * start_norm
    t1 = float(clip_duration) * max(start_norm + 1e-4, end_norm)
    if t1 - t0 < 1e-4:
        return

    intensity = _safe_float(effect.get("intensity", 0.5), 0.5, min_value=0.0, max_value=2.0)
    speed = _safe_float(effect.get("speed", 1.0), 1.0, min_value=0.05, max_value=20.0)
    seed = int(effect.get("seed", 0)) + int(clip_index)
    rng = np.random.default_rng(seed)

    x0, y0 = base_position
    sx0, sy0 = base_scale
    fx_dur = max(1e-3, t1 - t0)

    if kind == "shake":
        amp = min(width, height) * 0.006 * intensity
        steps = max(6, int(fx_dur * 18.0 * speed))
        times = np.linspace(t0, t1, steps)
        vals = [(float(x0 + rng.uniform(-amp, amp)), float(y0 + rng.uniform(-amp, amp))) for _ in range(steps - 1)] + [base_position]
        _extend_motion_safe(item, "position", [float(t) for t in times], vals)
        return

    if kind == "zoom_pulse":
        cycles = max(1, int(round(fx_dur * speed * 2.0)))
        times = np.linspace(t0, t1, cycles * 2 + 1)
        amp = 0.03 * intensity
        vals = []
        for i in range(len(times)):
            phase = np.sin((i / max(1, len(times) - 1)) * np.pi * cycles)
            k = 1.0 + amp * float(phase)
            vals.append((float(sx0 * k), float(sy0 * k)))
        vals[-1] = base_scale
        _extend_motion_safe(item, "scale", [float(t) for t in times], vals)
        return

    if kind == "strobe":
        hz = max(2.0, speed * 8.0)
        steps = max(6, int(fx_dur * hz))
        times = np.linspace(t0, t1, steps)
        low = max(0.0, base_opacity * (1.0 - 0.75 * intensity))
        vals = [float(base_opacity if i % 2 == 0 else low) for i in range(steps)]
        vals[-1] = float(base_opacity)
        _extend_motion_safe(item, "opacity", [float(t) for t in times], vals)
        return

    if kind == "drift":
        dx = width * 0.04 * intensity
        dy = height * 0.02 * intensity
        _extend_motion_safe(
            item,
            "position",
            [t0, t0 + fx_dur * 0.5, t1],
            [base_position, (x0 + dx, y0 - dy), base_position],
            easings=["ease_out", "ease_in"],
        )
        return

    if kind == "spin_wiggle":
        deg = 8.0 * intensity
        _extend_motion_safe(
            item,
            "rotation",
            [t0, t0 + fx_dur * 0.25, t0 + fx_dur * 0.5, t0 + fx_dur * 0.75, t1],
            [base_rotation, base_rotation + deg, base_rotation - deg, base_rotation + deg * 0.4, base_rotation],
            easings=["ease_out", "linear", "linear", "ease_in"],
        )
        return

    if kind in {"glitch", "custom"}:
        parsed = _parse_custom_curve(str(effect.get("custom_curve", "")))
        if parsed:
            ts, vs = parsed
            times = [t0 + fx_dur * float(t) for t in ts]
            max_amp = min(width, height) * 0.01 * intensity
            pos_vals = [
                (
                    float(x0 + (float(v) - 0.5) * 2.0 * max_amp),
                    float(y0 + (0.5 - float(v)) * 2.0 * max_amp),
                )
                for v in vs
            ]
            _extend_motion_safe(item, "position", times, pos_vals)
            return

        steps = max(8, int(fx_dur * speed * 20.0))
        times = np.linspace(t0, t1, steps)
        pos_amp = min(width, height) * 0.008 * intensity
        rot_amp = 3.5 * intensity
        op_min = max(0.0, base_opacity * (1.0 - 0.45 * intensity))

        pos_vals = []
        rot_vals = []
        op_vals = []
        for _ in range(steps):
            pos_vals.append((float(x0 + rng.uniform(-pos_amp, pos_amp)), float(y0 + rng.uniform(-pos_amp, pos_amp))))
            rot_vals.append(float(base_rotation + rng.uniform(-rot_amp, rot_amp)))
            op_vals.append(float(rng.uniform(op_min, base_opacity)))
        pos_vals[-1] = base_position
        rot_vals[-1] = base_rotation
        op_vals[-1] = base_opacity

        _extend_motion_safe(item, "position", [float(t) for t in times], pos_vals)
        _extend_motion_safe(item, "rotation", [float(t) for t in times], rot_vals)
        _extend_motion_safe(item, "opacity", [float(t) for t in times], op_vals)
        return


def _build_fx_preset(preset: str, strength: float, seed: int) -> list[dict[str, Any]]:
    p = str(preset or "cyber_glitch").strip().lower()
    s = _safe_float(strength, 0.8, min_value=0.05, max_value=2.0)
    if p == "impact_shake":
        return [
            _make_fx("shake", 0.0, 0.35, min(2.0, 1.3 * s), 1.5, seed),
            _make_fx("zoom_pulse", 0.0, 0.35, 0.8 * s, 1.6, seed + 1),
        ]
    if p == "dream_pulse":
        return [
            _make_fx("drift", 0.0, 1.0, 0.7 * s, 0.8, seed),
            _make_fx("zoom_pulse", 0.1, 0.95, 0.45 * s, 0.7, seed + 1),
        ]
    if p == "energy_spin":
        return [
            _make_fx("spin_wiggle", 0.0, 0.55, 1.0 * s, 1.2, seed),
            _make_fx("glitch", 0.55, 1.0, 0.6 * s, 1.3, seed + 1),
        ]
    # cyber_glitch (default)
    return [
        _make_fx("glitch", 0.0, 0.28, 0.95 * s, 1.6, seed),
        _make_fx("strobe", 0.28, 0.42, 0.55 * s, 1.1, seed + 1),
        _make_fx("zoom_pulse", 0.42, 1.0, 0.6 * s, 0.9, seed + 2),
    ]


def _transition_motion_values(kind: str, entering: bool, position: tuple[float, float], scale: tuple[float, float], width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y = position
    sx, sy = scale
    offx = width * 0.9
    offy = height * 0.9

    if kind in {"slide_left", "wipe_left", "push_left"}:
        start_pos = (x + offx, y)
        end_pos = (x, y)
        return (start_pos, end_pos) if entering else (end_pos, (x - offx, y))
    if kind in {"slide_right", "wipe_right", "push_right"}:
        start_pos = (x - offx, y)
        end_pos = (x, y)
        return (start_pos, end_pos) if entering else (end_pos, (x + offx, y))
    if kind == "slide_up":
        start_pos = (x, y + offy)
        end_pos = (x, y)
        return (start_pos, end_pos) if entering else (end_pos, (x, y - offy))
    if kind == "slide_down":
        start_pos = (x, y - offy)
        end_pos = (x, y)
        return (start_pos, end_pos) if entering else (end_pos, (x, y + offy))
    if kind == "zoom_in":
        start_scale = (sx * 1.2, sy * 1.2)
        end_scale = (sx, sy)
        return (start_scale, end_scale) if entering else (end_scale, (sx * 0.9, sy * 0.9))
    if kind == "zoom_out":
        start_scale = (sx * 0.9, sy * 0.9)
        end_scale = (sx, sy)
        return (start_scale, end_scale) if entering else (end_scale, (sx * 1.2, sy * 1.2))
    return (position, position) if kind not in {"zoom_in", "zoom_out"} else (scale, scale)


def _apply_transition(item, transition: dict[str, Any] | None, clip_duration: float, base_position: tuple[float, float], base_scale: tuple[float, float], base_opacity: float, width: int, height: int, entering: bool, allow_spatial: bool):
    if not transition:
        return
    kind = str(transition.get("type", "none"))
    if kind == "none":
        return
    dur = _safe_float(transition.get("duration", 0.0), 0.0, min_value=0.0)
    if dur <= 0:
        return
    d = min(float(clip_duration), dur)
    easing = str(transition.get("easing", "ease_in_out"))
    softness = _safe_float(transition.get("softness", 0.5), 0.5, min_value=0.0, max_value=1.0)
    custom_curve = str(transition.get("custom_curve", ""))

    if entering:
        t0, t1 = 0.0, d
        op_from, op_to = (0.0, base_opacity)
    else:
        t0, t1 = max(0.0, float(clip_duration) - d), float(clip_duration)
        op_from, op_to = (base_opacity, 0.0)

    if kind == "custom":
        parsed = _parse_custom_curve(custom_curve)
        if parsed:
            ts, vs = parsed
            kfs = [t0 + (t1 - t0) * t for t in ts]
            vals = [float(base_opacity) * float(v) for v in vs]
            _extend_motion_safe(item, "opacity", kfs, vals)
            return
        kind = "crossfade"

    # opacity curve for all transitions
    if kind in {"fade_black", "crossfade", "wipe_left", "wipe_right", "push_left", "push_right", "slide_left", "slide_right", "slide_up", "slide_down", "zoom_in", "zoom_out"}:
        mid = t0 + (t1 - t0) * (0.5 + softness * 0.15)
        if easing == "linear":
            _extend_motion_safe(item, "opacity", [t0, t1], [op_from, op_to])
        elif easing == "ease_in":
            _extend_motion_safe(item, "opacity", [t0, mid, t1], [op_from, op_from + (op_to - op_from) * 0.35, op_to])
        elif easing == "ease_out":
            _extend_motion_safe(item, "opacity", [t0, mid, t1], [op_from, op_from + (op_to - op_from) * 0.75, op_to])
        else:
            _extend_motion_safe(item, "opacity", [t0, mid, t1], [op_from, op_from + (op_to - op_from) * 0.5, op_to])

    if not allow_spatial:
        return

    if kind in {"slide_left", "slide_right", "slide_up", "slide_down", "wipe_left", "wipe_right", "push_left", "push_right"}:
        p0, p1 = _transition_motion_values(kind, entering, base_position, base_scale, width, height)
        _extend_motion_safe(item, "position", [t0, t1], [p0, p1], easings=[easing])
    elif kind in {"zoom_in", "zoom_out"}:
        s0, s1 = _transition_motion_values(kind, entering, base_position, base_scale, width, height)
        _extend_motion_safe(item, "scale", [t0, t1], [s0, s1], easings=[easing])


def _attach_fx_shader_to_clip(clip: dict, fx=None, shader=None) -> dict:
    """Apply optional MOVIS_FX list and MOVIS_SHADER list to a clip dict in-place.

    MOVIS_FX items with ``_engine == "layered"`` go into ``advanced_fx_layers``;
    all others go into ``effects``.  MOVIS_SHADER items go into ``clip_shaders``.
    """
    if fx and isinstance(fx, list):
        for item in fx:
            if not isinstance(item, dict):
                continue
            item_clean = {k: v for k, v in item.items() if k != "_engine"}
            if item.get("_engine") == "layered":
                existing = clip.setdefault("advanced_fx_layers", [])
                existing.append(item_clean)
                clip["use_layered_fx_engine"] = True
            else:
                existing = clip.setdefault("effects", [])
                existing.append(item_clean)
    if shader and isinstance(shader, list):
        existing = clip.setdefault("clip_shaders", [])
        for item in shader:
            if isinstance(item, dict):
                existing.append(item)
    return clip


def _attach_audio_to_clip(timeline: dict, clip_start: float, audio_list) -> None:
    """Attach MOVIS_AUDIO entries to the timeline, anchored at clip_start.

    Each MOVIS_AUDIO entry is a dict::

        {
            "path": str,
            "duration": float,        # 0 = auto-detect from file
            "source_start": float,    # start offset within the source file
            "audio_level_db": float,
            "offset": float,          # seconds after clip_start; 0 = align with clip
        }

    Multiple entries on a single clip enable genuine multi-audio-layer support:
    e.g. dialogue + SFX + stems all pinned to the same video clip.
    """
    if not audio_list or not isinstance(audio_list, list):
        return
    for entry in audio_list:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        if not path or not os.path.exists(path):
            continue
        dur = float(entry.get("duration", 0.0))
        if dur <= 0:
            dur = _audio_duration(path)
        resolved_start = max(0.0, clip_start + float(entry.get("offset", 0.0)))
        timeline["audio_tracks"].append({
            "path": path,
            "start": resolved_start,
            "duration": max(0.01, dur),
            "source_start": float(entry.get("source_start", 0.0)),
            "audio_level_db": float(entry.get("audio_level_db", 0.0)),
        })


def _smart_bgm_level(timeline: dict[str, Any], default_bgm_db: float = -12.0, duck_db: float = -18.0) -> float:
    """若 auto_duck_bgm=True 且存在前景音轨，则自动压低 BGM。"""
    bgm = timeline.get("bgm") or {}
    base = float(bgm.get("audio_level_db", default_bgm_db))
    auto_duck = bool(bgm.get("auto_duck_bgm", False))
    if auto_duck:
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
    _bgm_meta = timeline.get("bgm") or {}
    _trim_bgm = bool(_bgm_meta.get("trim_to_video_length", True)) if _bgm_meta.get("path") else False
    if _trim_bgm:
        # 语义：trim_to_video_length 应严格以“视频内容时长”为准，避免 BGM/字幕把时长拉长导致尾部黑屏。
        _video_dur = _timeline_video_duration(timeline)
        _fallback_dur = _timeline_content_duration(timeline)
        duration = _safe_float(_video_dur if _video_dur > 0.01 else _fallback_dur, 1.0, min_value=0.01)
    else:
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

    clips = timeline.get("video_tracks", [])
    total_clips = len(clips)
    for clip_index, clip in enumerate(clips):
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

        use_advanced_engine = bool(clip.get("use_layered_fx_engine", False)) or (
            isinstance(clip.get("advanced_fx_layers"), list) and len(clip.get("advanced_fx_layers")) > 0
        )

        if use_advanced_engine:
            _apply_advanced_clip_animation(
                item,
                timeline,
                clip,
                clip_duration,
                position,
                scale,
                rotation,
                opacity,
                width,
                height,
                fps,
                clip_index,
                total_clips,
            )
            pbar.update(1)
            continue


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
                _extend_motion_safe(item, "position", motion_times, pos_values, easings=easings, clear=True)

            if sx or sy:
                scale_values = []
                for i in range(len(motion_times)):
                    x = _safe_float(sx[i], scale[0], min_value=0.01, max_value=20.0) if i < len(sx) else scale[0]
                    y = _safe_float(sy[i], scale[1], min_value=0.01, max_value=20.0) if i < len(sy) else scale[1]
                    scale_values.append((x, y))
                _extend_motion_safe(item, "scale", motion_times, scale_values, easings=easings, clear=True)

            if rt:
                rot_values = []
                for i in range(len(motion_times)):
                    rot_values.append(_safe_float(rt[i], rotation, min_value=-360.0, max_value=360.0) if i < len(rt) else rotation)
                _extend_motion_safe(item, "rotation", motion_times, rot_values, easings=easings, clear=True)

            if op:
                opacity_values = []
                for i in range(len(motion_times)):
                    opacity_values.append(_safe_float(op[i], opacity, min_value=0.0, max_value=1.0) if i < len(op) else opacity)
                _extend_motion_safe(item, "opacity", motion_times, opacity_values, easings=easings, clear=True)
                motion_applied_opacity = True

        transition_in = _resolve_clip_transition(timeline, clip, clip_index, total_clips, entering=True)
        transition_out = _resolve_clip_transition(timeline, clip, clip_index, total_clips, entering=False)

        if transition_in is not None:
            _apply_transition(
                item,
                transition_in,
                clip_duration,
                position,
                scale,
                opacity,
                width,
                height,
                entering=True,
                allow_spatial=not bool(clip.get("is_image", False)),
            )
        elif fade_in > 0 and not motion_applied_opacity:
            _extend_motion_safe(item, "opacity", [0.0, fade_in], [0.0, opacity])

        if transition_out is not None:
            _apply_transition(
                item,
                transition_out,
                clip_duration,
                position,
                scale,
                opacity,
                width,
                height,
                entering=False,
                allow_spatial=not bool(clip.get("is_image", False)),
            )
        elif fade_out > 0 and not motion_applied_opacity:
            t0 = max(0.0, clip_duration - fade_out)
            _extend_motion_safe(item, "opacity", [t0, clip_duration], [opacity, 0.0])

        for fx in _iter_clip_effects(clip):
            _apply_clip_fx(
                item,
                fx,
                clip_duration,
                position,
                scale,
                rotation,
                opacity,
                width,
                height,
                clip_index,
            )
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
        safe_family = validate_font_family(str(txt.get("font_family", "Sans Serif")))
        safe_style = validate_font_style(safe_family, str(txt.get("font_style", "Regular")))
        layer = _create_movis_text_layer(
            text=str(txt.get("text", "")),
            font_size=float(txt.get("font_size", 64)),
            font_family=safe_family,
            font_style=safe_style,
            color=str(txt.get("color", "white")),
            duration=float(txt.get("duration", 0.01)),
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
    # Choose GPU encoder (h264_nvenc) when available, fall back to CPU libx264
    _movis_codec = "h264_nvenc" if _check_nvenc_available() else "libx264"
    print(f"[VHS_MOVIS] video codec: {_movis_codec}")
    try:
        scene.write_video(
            out_path,
            codec=_movis_codec,
            pixelformat="yuv420p",
            fps=fps,
            audio=True,
            output_params=["-movflags", "+faststart"],
        )
    except Exception as _enc_err:
        if _movis_codec != "libx264":
            print(f"[VHS_MOVIS] {_movis_codec} failed ({_enc_err}), retrying with libx264 ...")
            scene.write_video(
                out_path,
                codec="libx264",
                pixelformat="yuv420p",
                fps=fps,
                audio=True,
                output_params=["-movflags", "+faststart"],
            )
        else:
            raise
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
                "transition_in": (TRANSITION_TYPES, {"default": "none", "tooltip": "片段入场转场"}),
                "transition_in_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_out": (TRANSITION_TYPES, {"default": "none", "tooltip": "片段出场转场"}),
                "transition_out_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
                "transition_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "transition_custom_curve": ("STRING", {"default": "", "placeholder": "仅custom使用: 0:0,0.3:0.2,1:1"}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
                "fx": ("MOVIS_FX", {"tooltip": "可选：MOVIS_FX特效，自动绑定到本片段，无需设置clip_index"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：MOVIS_SHADER，自动绑定到本片段"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：MOVIS_AUDIO音轨，自动与本片段对齐；支持多轨叠加（对话+音效+伴奏）"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_video"

    def add_video(self, timeline, video_path, placement_mode, start, duration, source_start, fade_in, fade_out, use_source_audio, audio_level_db, position_x, position_y, scale_x, scale_y, rotation, opacity, transition_in, transition_in_duration, transition_out, transition_out_duration, transition_easing, transition_softness, transition_custom_curve, vhs_filenames=None, fx=None, shader=None, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_video_input(video_path, vhs_filenames=vhs_filenames)
        clip_duration = _safe_float(duration, 0.0, min_value=0.0)
        if clip_duration <= 0:
            clip_duration = _video_duration(path)
        start_base = _timeline_content_duration(t) if str(placement_mode) == "append" else 0.0
        resolved_start = start_base + _safe_float(start, 0.0, min_value=0.0)
        clip = {
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
                "transition_in": _make_transition(transition_in, transition_in_duration, transition_easing, transition_softness, transition_custom_curve),
                "transition_out": _make_transition(transition_out, transition_out_duration, transition_easing, transition_softness, transition_custom_curve),
        }
        _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
        _attach_audio_to_clip(t, resolved_start, audio)
        t["video_tracks"].append(clip)
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
                "transition_in": (TRANSITION_TYPES, {"default": "none", "tooltip": "片段入场转场"}),
                "transition_in_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_out": (TRANSITION_TYPES, {"default": "none", "tooltip": "片段出场转场"}),
                "transition_out_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
                "transition_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "transition_custom_curve": ("STRING", {"default": "", "placeholder": "仅custom使用: 0:0,0.3:0.2,1:1"}),
                "image_scale_mode": (["transform_only", "contain", "cover", "center_inside", "stretch"], {"default": "transform_only", "tooltip": "图片适配画布策略"}),
                "image_anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "水平锚点；cover/contain时决定裁剪或摆放焦点"}),
                "image_anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "tooltip": "垂直锚点；cover/contain时决定裁剪或摆放焦点"}),
                "image_bg_color": ("STRING", {"default": "#00000000", "tooltip": "contain/center_inside时的填充背景色"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "fx": ("MOVIS_FX", {"tooltip": "可选：MOVIS_FX特效，自动绑定到本片段"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：MOVIS_SHADER，自动绑定到本片段"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：MOVIS_AUDIO音轨，自动与本图片片段对齐"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_image"

    def add_image(self, timeline, image_path, start, duration, fade_in, fade_out, position_x, position_y, scale_x, scale_y, rotation, opacity, transition_in, transition_in_duration, transition_out, transition_out_duration, transition_easing, transition_softness, transition_custom_curve, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color, image=None, fx=None, shader=None, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_image_input(image_path, image=image, prefix="movis_add_image")
        clip = {
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
                "transition_in": _make_transition(transition_in, transition_in_duration, transition_easing, transition_softness, transition_custom_curve),
                "transition_out": _make_transition(transition_out, transition_out_duration, transition_easing, transition_softness, transition_custom_curve),
                "image_scale_mode": str(image_scale_mode),
                "image_anchor_x": _safe_float(image_anchor_x, 0.5, min_value=0.0, max_value=1.0),
                "image_anchor_y": _safe_float(image_anchor_y, 0.5, min_value=0.0, max_value=1.0),
                "image_bg_color": str(image_bg_color),
        }
        _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
        _attach_audio_to_clip(t, _safe_float(start, 0.0, min_value=0.0), audio)
        t["video_tracks"].append(clip)
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
            },
            "optional": {
                "fx": ("MOVIS_FX", {"tooltip": "可选：MOVIS_FX特效，应用到序列内所有图片片段"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：MOVIS_SHADER，应用到序列内所有图片片段"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：MOVIS_AUDIO音轨，与整个图片序列的起始时间对齐"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE", "INT", "FLOAT")
    RETURN_NAMES = ("timeline", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_images"

    def add_images(self, timeline, images, start, seconds_per_image, durations_csv, fade_in, fade_out, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color, fx=None, shader=None, audio=None):
        t = _clone_timeline(timeline)
        image_files = _save_images(images, f"movis_img_seq_{_now_stamp()}")
        durations = _parse_float_list(durations_csv)
        cursor = float(start)
        total_duration = 0.0
        for idx, p in enumerate(image_files):
            dur = float(seconds_per_image)
            if idx < len(durations):
                dur = max(0.01, float(durations[idx]))
            clip = {
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
            _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
            t["video_tracks"].append(clip)
            cursor += dur
            total_duration += dur
        fps = float(t["canvas"]["fps"])
        _attach_audio_to_clip(t, float(start), audio)
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
                "fx": ("MOVIS_FX", {"tooltip": "可选：MOVIS_FX特效，自动绑定到本片段"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：MOVIS_SHADER，自动绑定到本片段"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：MOVIS_AUDIO音轨，与本片段对齐"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_motion"

    def add_motion(self, timeline, image_path, start, duration, keyframe_times_csv, position_x_csv, position_y_csv, scale_x_csv, scale_y_csv, rotation_csv, opacity_csv, easing_csv, fade_in, fade_out, image_scale_mode, image_anchor_x, image_anchor_y, image_bg_color, image=None, fx=None, shader=None, audio=None):
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

        clip = {
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
        _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
        _attach_audio_to_clip(t, float(start), audio)
        t["video_tracks"].append(clip)
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
                "fx": ("MOVIS_FX", {"tooltip": "可选：MOVIS_FX特效，自动绑定到本片段"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：MOVIS_SHADER，自动绑定到本片段"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：MOVIS_AUDIO音轨，与本片段对齐"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_video_motion"

    def add_video_motion(self, timeline, video_path, start, duration, source_start, keyframe_times_csv, position_x_csv, position_y_csv, scale_x_csv, scale_y_csv, rotation_csv, opacity_csv, easing_csv, fade_in, fade_out, use_source_audio, audio_level_db, vhs_filenames=None, fx=None, shader=None, audio=None):
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

        clip = {
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
        _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
        _attach_audio_to_clip(t, float(start), audio)
        t["video_tracks"].append(clip)
        return (t,)

class MovisAddAudioTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "audio_path": ("STRING", {"default": "", "placeholder": "可留空，改接 AUDIO"}),
                "placement_mode": (["match_last_video", "append", "absolute"], {"default": "match_last_video", "tooltip": "match_last_video: 对齐最近视频片段起点；append: 追加到时间线末尾；absolute: 使用 start 绝对时间"}),
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

    def add_audio(self, timeline, audio_path, placement_mode, start, duration, source_start, audio_level_db, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_audio_input(audio_path, audio=audio, prefix="movis_audio_track")
        clip_duration = _safe_float(duration, 0.0, min_value=0.0)
        if clip_duration <= 0:
            clip_duration = _audio_duration(path)

        mode = str(placement_mode or "match_last_video").strip().lower()
        start_value = _safe_float(start, 0.0, min_value=0.0)
        if mode == "append":
            resolved_start = _timeline_content_duration(t) + start_value
        elif mode == "match_last_video":
            videos = t.get("video_tracks", [])
            if isinstance(videos, list) and len(videos) > 0:
                audios = t.get("audio_tracks", [])
                # 优先按“第N条音轨对应第N个视频片段”对齐，便于一段视频配一段音频。
                # 若音轨数量已超过视频数量，则回退到最后一个视频片段。
                if isinstance(audios, list) and len(audios) < len(videos):
                    target_video = videos[len(audios)]
                else:
                    target_video = videos[-1]
                resolved_start = _safe_float(target_video.get("start", 0.0), 0.0, min_value=0.0) + start_value
            else:
                resolved_start = start_value
        else:
            resolved_start = start_value

        t["audio_tracks"].append(
            {
                "path": path,
                "start": resolved_start,
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
                "bgm_path": ("STRING", {"default": "", "placeholder": "可留空，改接 AUDIO", "tooltip": "全局BGM，仅保留一条，会覆盖此前设置"}),
                "audio_level_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 40.0, "tooltip": "BGM增益(dB)；已提高上限至40dB，如仍感觉偏小可继续调高"}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "trim_to_video_length": ("BOOLEAN", {"default": True, "tooltip": "BGM超出视频内容总时长时自动截断，防止末尾黑屏；关闭则以BGM文件原长为准"}),
                "auto_duck_bgm": ("BOOLEAN", {"default": False, "tooltip": "检测到前景音轨(视频自带音/外部音轨)时自动将BGM压低至-18dB；默认关闭，需要时手动开启"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_bgm"

    def set_bgm(self, timeline, bgm_path, audio_level_db, source_start, trim_to_video_length=True, auto_duck_bgm=False, audio=None):
        t = _clone_timeline(timeline)
        path = _resolve_audio_input(bgm_path, audio=audio, prefix="movis_bgm")
        t["bgm"] = {
            "path": path,
            "audio_level_db": _safe_float(audio_level_db, -12.0, min_value=-60.0, max_value=40.0),
            "source_start": _safe_float(source_start, 0.0, min_value=0.0),
            "duration": _audio_duration(path),
            "trim_to_video_length": bool(trim_to_video_length),
            "auto_duck_bgm": bool(auto_duck_bgm),
        }
        return (t,)


class MovisAddTextOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = get_movis_fonts()
        default_font = choose_default_font(fonts)
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "text": ("STRING", {"default": "字幕文本", "multiline": True}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "duration": ("FLOAT", {"default": 2.0, "min": 0.01}),
                "font_size": ("FLOAT", {"default": 64.0, "min": 8.0, "max": 300.0}),
                "font_family": (fonts, {"default": default_font}),
                "font_style": (_COMMON_FONT_STYLES, {"default": "Regular"}),
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

    def add_text(self, timeline, text, start, duration, font_size, font_family, font_style="Regular", color="white", position_x=0.5, position_y=0.9, opacity=1.0):
        t = _clone_timeline(timeline)
        safe_family = validate_font_family(font_family)
        safe_style = validate_font_style(safe_family, font_style)
        t["text_tracks"].append(
            {
                "text": text,
                "start": _safe_float(start, 0.0, min_value=0.0),
                "duration": _safe_float(duration, 2.0, min_value=0.01),
                "font_size": _safe_float(font_size, 64.0, min_value=8.0, max_value=300.0),
                "font_family": safe_family,
                "font_style": safe_style,
                "color": color,
                "position_x": _safe_float(position_x, 0.5),
                "position_y": _safe_float(position_y, 0.9),
                "opacity": _safe_float(opacity, 1.0, min_value=0.0, max_value=1.0),
            }
        )
        return (t,)


class MovisRefreshFonts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rebuild_system_font_cache": ("BOOLEAN", {"default": True}),
                "clear_plugin_cache": ("BOOLEAN", {"default": True}),
                "refresh_movis_cache": ("BOOLEAN", {"default": True}),
                "output_limit": ("INT", {"default": 300, "min": 1, "max": 5000}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "INT")
    RETURN_NAMES = ("font_count", "fonts_text", "cache_version")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "refresh"

    def refresh(self, rebuild_system_font_cache, clear_plugin_cache, refresh_movis_cache, output_limit):
        if bool(clear_plugin_cache):
            clear_movis_font_caches()

        if sys.platform.startswith("linux") and bool(rebuild_system_font_cache):
            try:
                subprocess.run(
                    ["fc-cache", "-fv"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception as e:
                print(f"[Movis Font] fc-cache execution skipped/failed: {e}")

        fonts = get_movis_fonts(refresh=bool(refresh_movis_cache))
        limit = _clamp_int(output_limit, 300, 1, 5000)
        text_items = fonts[:limit]
        fonts_text = "\n".join(text_items)
        if len(fonts) > limit:
            fonts_text += f"\n... ({len(fonts) - limit} more fonts omitted)"
        return (len(fonts), fonts_text, get_movis_font_cache_version())


class MovisFontPreview:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = get_movis_fonts()
        default_font = choose_default_font(fonts)
        return {
            "required": {
                "text": ("STRING", {"default": "Lemon 字体预览 123 ABC", "multiline": True}),
                "font_family": (fonts, {"default": default_font}),
                "font_style": (_COMMON_FONT_STYLES, {"default": "Regular"}),
                "font_size": ("INT", {"default": 64, "min": 8, "max": 512}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "bg_color": ("STRING", {"default": "#222222"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 200, "min": 64, "max": 2048}),
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "align": (["left", "center", "right"], {"default": "center"}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 128}),
                "stroke_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "preview"

    def preview(self, text, font_family, font_style, font_size, text_color, bg_color, width, height, x, y, align, stroke_width, stroke_color):
        image, _meta = render_font_preview_image(
            font_family=font_family,
            font_style=font_style,
            text=text,
            font_size=font_size,
            width=width,
            height=height,
            text_color=text_color,
            bg_color=bg_color,
            x=x,
            y=y,
            align=align,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
        )
        return (_pil_to_comfy_image(image.convert("RGB")),)


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


class MovisSetGlobalTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "transition": (TRANSITION_TYPES, {"default": "crossfade", "tooltip": "全局转场类型"}),
                "duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
                "softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "custom_curve": ("STRING", {"default": "", "placeholder": "仅custom使用: 0:0,0.3:0.2,1:1"}),
                "apply_to_first": ("BOOLEAN", {"default": False}),
                "apply_to_last": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_global_transition"

    def set_global_transition(self, timeline, transition, duration, easing, softness, custom_curve, apply_to_first, apply_to_last):
        t = _clone_timeline(timeline)
        tr = _make_transition(transition, duration, easing, softness, custom_curve)
        if tr is None:
            t.pop("global_transition", None)
        else:
            tr["apply_to_first"] = bool(apply_to_first)
            tr["apply_to_last"] = bool(apply_to_last)
            t["global_transition"] = tr
        return (t,)


class MovisSetClipTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999}),
                "set_in": ("BOOLEAN", {"default": True}),
                "transition_in": (TRANSITION_TYPES, {"default": "crossfade"}),
                "transition_in_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "set_out": ("BOOLEAN", {"default": True}),
                "transition_out": (TRANSITION_TYPES, {"default": "crossfade"}),
                "transition_out_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
                "softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "custom_curve": ("STRING", {"default": "", "placeholder": "仅custom使用: 0:0,0.3:0.2,1:1"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_clip_transition"

    def set_clip_transition(self, timeline, clip_index, set_in, transition_in, transition_in_duration, set_out, transition_out, transition_out_duration, easing, softness, custom_curve):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        if set_in:
            clips[idx]["transition_in"] = _make_transition(transition_in, transition_in_duration, easing, softness, custom_curve)
        if set_out:
            clips[idx]["transition_out"] = _make_transition(transition_out, transition_out_duration, easing, softness, custom_curve)
        return (t,)


class MovisAddClipFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "effect": (FX_TYPES, {"default": "glitch", "tooltip": "特效类型（近似 shader 风格）"}),
                "start_norm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "tooltip": "特效开始时间（片段归一化）"}),
                "end_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "特效结束时间（片段归一化）"}),
                "intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "tooltip": "特效强度"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0, "tooltip": "特效速度"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "replace_existing": ("BOOLEAN", {"default": False, "tooltip": "是否替换该片段已有特效"}),
                "custom_curve": ("STRING", {"default": "", "placeholder": "仅 custom 推荐：0:0.5,0.2:1,0.8:0,1:0.5"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_fx"

    def add_fx(self, timeline, clip_index, effect, start_norm, end_norm, intensity, speed, seed, replace_existing, custom_curve):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        fx = _make_fx(effect, start_norm, end_norm, intensity, speed, seed, custom_curve)
        if fx is None:
            return (t,)
        if bool(replace_existing):
            clips[idx]["effects"] = [fx]
        else:
            existing = clips[idx].get("effects")
            if not isinstance(existing, list):
                existing = []
            existing.append(fx)
            clips[idx]["effects"] = existing
        return (t,)


class MovisApplyFXPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "preset": (FX_PRESETS, {"default": "cyber_glitch", "tooltip": "预设：一键酷炫效果"}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.05, "max": 2.0, "tooltip": "预设强度"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "replace_existing": ("BOOLEAN", {"default": False, "tooltip": "是否替换已有特效"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "apply_preset"

    def apply_preset(self, timeline, clip_index, preset, strength, seed, replace_existing):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        fx_list = [x for x in _build_fx_preset(preset, strength, seed) if isinstance(x, dict)]
        if len(fx_list) == 0:
            return (t,)
        if bool(replace_existing):
            clips[idx]["effects"] = fx_list
        else:
            existing = clips[idx].get("effects")
            if not isinstance(existing, list):
                existing = []
            existing.extend(fx_list)
            clips[idx]["effects"] = existing
        return (t,)



class MovisEnableLayeredFXEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "启用更高级的分层动画引擎；原有节点保留不变"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_engine"

    def set_engine(self, timeline, clip_index, enabled):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        clips[idx]["use_layered_fx_engine"] = bool(enabled)
        return (t,)


class MovisAddClipFXLayered:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "effect": (FX_TYPES, {"default": "glitch", "tooltip": "分层特效：可与 motion/transition 同时叠加而不抢 keyframe"}),
                "start_norm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "replace_existing": ("BOOLEAN", {"default": False}),
                "custom_curve": ("STRING", {"default": "", "placeholder": "仅 custom 推荐：0:0.5,0.2:1,0.8:0,1:0.5"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "add_fx_layered"

    def add_fx_layered(self, timeline, clip_index, effect, start_norm, end_norm, intensity, speed, seed, replace_existing, custom_curve):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        fx = _make_fx(effect, start_norm, end_norm, intensity, speed, seed, custom_curve)
        if fx is None:
            return (t,)
        existing = clips[idx].get("advanced_fx_layers")
        if not isinstance(existing, list) or bool(replace_existing):
            existing = []
        existing.append(fx)
        clips[idx]["advanced_fx_layers"] = existing
        clips[idx]["use_layered_fx_engine"] = True
        return (t,)


class MovisApplyFXPresetLayered:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "preset": (FX_PRESETS, {"default": "cyber_glitch"}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.05, "max": 2.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "replace_existing": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "apply_preset_layered"

    def apply_preset_layered(self, timeline, clip_index, preset, strength, seed, replace_existing):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        fx_list = [x for x in _build_fx_preset(preset, strength, seed) if isinstance(x, dict)]
        if len(fx_list) == 0:
            return (t,)
        existing = clips[idx].get("advanced_fx_layers")
        if not isinstance(existing, list) or bool(replace_existing):
            existing = []
        existing.extend(fx_list)
        clips[idx]["advanced_fx_layers"] = existing
        clips[idx]["use_layered_fx_engine"] = True
        return (t,)


class MovisSetClipShader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999, "tooltip": "目标视频片段索引，-1 表示最后一段"}),
                "shader_path": ("STRING", {"default": "", "placeholder": "GLSL shader 文件路径（可空）"}),
                "shader_code": ("STRING", {"default": "", "multiline": True, "placeholder": "可直接粘贴 GLSL 代码；非空时优先于 shader_path"}),
                "backend": (GPU_SHADER_BACKENDS, {"default": "libplacebo_vulkan"}),
                "codec": (GPU_SHADER_CODECS, {"default": "libx264"}),
                "keep_audio": ("BOOLEAN", {"default": True, "tooltip": "clip 预处理时是否保留原音频"}),
                "replace_existing": ("BOOLEAN", {"default": False, "tooltip": "是否替换该片段已有 clip shader"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "关闭后保留配置但本次不生效"}),
                "output_file_prefix": ("STRING", {"default": "movis_clip_shader_", "tooltip": "clip shader 临时输出前缀"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "set_clip_shader"

    def set_clip_shader(self, timeline, clip_index, shader_path, shader_code, backend, codec, keep_audio, replace_existing, enabled, output_file_prefix):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        idx = _resolve_video_clip_index(clips, clip_index)
        if idx is None:
            return (t,)
        shader = _normalize_clip_shader_entry({
            "shader_path": shader_path,
            "shader_code": shader_code,
            "backend": backend,
            "codec": codec,
            "keep_audio": keep_audio,
            "enabled": enabled,
            "output_file_prefix": output_file_prefix,
        })
        if shader is None:
            return (t,)
        existing = clips[idx].get("clip_shaders")
        if not isinstance(existing, list) or bool(replace_existing):
            existing = []
        existing.append(shader)
        clips[idx]["clip_shaders"] = existing
        return (t,)


class MovisGPUShaderRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "placeholder": "可留空，改接 VHS_FILENAMES"}),
                "shader_path": ("STRING", {"default": "", "placeholder": "GLSL shader 文件路径（可空）"}),
                "shader_code": ("STRING", {"default": "", "multiline": True, "placeholder": "可直接粘贴 GLSL 代码；非空时优先于 shader_path"}),
                "backend": (GPU_SHADER_BACKENDS, {"default": "libplacebo_vulkan"}),
                "codec": (GPU_SHADER_CODECS, {"default": "libx264"}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "output_file_prefix": ("STRING", {"default": "movis_gpu_shader_"}),
                "notify_all": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("filenames", "video_path", "audio", "frames", "duration")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "render_shader"
    OUTPUT_NODE = True

    def render_shader(self, video_path, shader_path, shader_code, backend, codec, keep_audio, output_file_prefix, notify_all, vhs_filenames=None):
        src_video = _resolve_video_input(video_path, vhs_filenames=vhs_filenames)
        shader_file = _resolve_shader_file(shader_path, shader_code, prefix="movis_gpu_shader")
        out_path = _apply_gpu_shader_ffmpeg(
            input_video_path=src_video,
            shader_file_path=shader_file,
            output_file_prefix=output_file_prefix,
            backend=backend,
            codec=codec,
            keep_audio=bool(keep_audio),
        )
        duration = _safe_float(_video_duration(out_path), 0.01, min_value=0.01)
        frames = max(1, int(round(duration * 30.0)))
        try:
            audio = get_audio(out_path)
        except Exception:
            audio = {
                "waveform": torch.zeros((1, 2, 0), dtype=torch.float32),
                "sample_rate": 44100,
            }
        if notify_all and notifyAll:
            notifyAll(out_path, "movis_gpu_shader")
        return {
            "ui": {"video_path": out_path, "frames": [frames]},
            "result": ((True, [out_path]), out_path, audio, frames, duration),
        }


class MovisTrimClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999}),
                "new_source_start": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "new_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "0 表示保持原时长"}),
                "shift_following": ("BOOLEAN", {"default": False, "tooltip": "是否整体平移后续片段保持无缝"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "trim_clip"

    def trim_clip(self, timeline, clip_index, new_source_start, new_duration, shift_following):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        if len(clips) == 0:
            return (t,)
        idx = int(clip_index)
        if idx < 0:
            idx = len(clips) + idx
        idx = max(0, min(len(clips) - 1, idx))

        old_duration = _safe_float(clips[idx].get("duration", 0.01), 0.01, min_value=0.01)
        clips[idx]["source_start"] = _safe_float(new_source_start, 0.0, min_value=0.0)
        if _safe_float(new_duration, 0.0, min_value=0.0) > 0:
            clips[idx]["duration"] = _safe_float(new_duration, old_duration, min_value=0.01)
        new_d = _safe_float(clips[idx].get("duration", old_duration), old_duration, min_value=0.01)
        delta = new_d - old_duration
        if shift_following and abs(delta) > 1e-8:
            base_start = _safe_float(clips[idx].get("start", 0.0), 0.0)
            for i, c in enumerate(clips):
                if i != idx and _safe_float(c.get("start", 0.0), 0.0) > base_start:
                    c["start"] = _safe_float(c.get("start", 0.0), 0.0) + delta
        return (t,)


class MovisDeleteClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "clip_index": ("INT", {"default": -1, "min": -99999, "max": 99999}),
                "compact_timeline": ("BOOLEAN", {"default": False, "tooltip": "删除后是否将后续片段左移填补空隙"}),
            }
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "delete_clip"

    def delete_clip(self, timeline, clip_index, compact_timeline):
        t = _clone_timeline(timeline)
        clips = t.get("video_tracks", [])
        if len(clips) == 0:
            return (t,)
        idx = int(clip_index)
        if idx < 0:
            idx = len(clips) + idx
        idx = max(0, min(len(clips) - 1, idx))
        removed = clips.pop(idx)
        if compact_timeline:
            removed_start = _safe_float(removed.get("start", 0.0), 0.0)
            removed_duration = _safe_float(removed.get("duration", 0.0), 0.0, min_value=0.0)
            for c in clips:
                cs = _safe_float(c.get("start", 0.0), 0.0)
                if cs >= removed_start:
                    c["start"] = cs - removed_duration
        return (t,)


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


# ---------------------------------------------------------------------------
# MOVIS_FX / MOVIS_SHADER builder nodes
# These nodes produce standalone FX/shader config objects that can be wired
# directly into any track-adding node's optional fx / shader input, so you
# never need to manually set clip_index.
# ---------------------------------------------------------------------------

class MovisBuildFX:
    """Build a MOVIS_FX config and wire it into any Add*Track node's optional fx input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "effect": (FX_TYPES, {"default": "glitch", "tooltip": "特效类型"}),
                "start_norm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "tooltip": "特效开始时间（片段归一化 0~1）"}),
                "end_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "特效结束时间（片段归一化 0~1）"}),
                "intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "engine": (["basic", "layered"], {"default": "basic", "tooltip": "basic: 普通特效引擎；layered: 分层特效引擎（可与动画/转场同时叠加）"}),
                "custom_curve": ("STRING", {"default": "", "placeholder": "仅 custom 推荐：0:0.5,0.2:1,0.8:0,1:0.5"}),
            },
            "optional": {
                "chain": ("MOVIS_FX", {"tooltip": "可选：将本特效追加到已有 MOVIS_FX 列表末尾"}),
            },
        }

    RETURN_TYPES = ("MOVIS_FX",)
    RETURN_NAMES = ("fx",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "build_fx"

    def build_fx(self, effect, start_norm, end_norm, intensity, speed, seed, engine, custom_curve, chain=None):
        fx = _make_fx(effect, start_norm, end_norm, intensity, speed, seed, custom_curve)
        if fx is None:
            return (list(chain) if chain else [],)
        fx["_engine"] = str(engine)
        result = list(chain) if isinstance(chain, list) else []
        result.append(fx)
        return (result,)


class MovisBuildFXPreset:
    """Build a MOVIS_FX list from a preset and wire it into any Add*Track node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (FX_PRESETS, {"default": "cyber_glitch", "tooltip": "预设：一键酷炫效果"}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.05, "max": 2.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "engine": (["basic", "layered"], {"default": "basic"}),
            },
            "optional": {
                "chain": ("MOVIS_FX", {"tooltip": "可选：将本预设追加到已有 MOVIS_FX 列表末尾"}),
            },
        }

    RETURN_TYPES = ("MOVIS_FX",)
    RETURN_NAMES = ("fx",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "build_preset"

    def build_preset(self, preset, strength, seed, engine, chain=None):
        fx_list = [x for x in _build_fx_preset(preset, strength, seed) if isinstance(x, dict)]
        for item in fx_list:
            item["_engine"] = str(engine)
        result = list(chain) if isinstance(chain, list) else []
        result.extend(fx_list)
        return (result,)


class MovisChainFX:
    """Merge two MOVIS_FX lists into one (for stacking multiple FX on the same clip)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fx_a": ("MOVIS_FX",),
            },
            "optional": {
                "fx_b": ("MOVIS_FX",),
                "fx_c": ("MOVIS_FX",),
            },
        }

    RETURN_TYPES = ("MOVIS_FX",)
    RETURN_NAMES = ("fx",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "chain_fx"

    def chain_fx(self, fx_a, fx_b=None, fx_c=None):
        result = list(fx_a) if isinstance(fx_a, list) else []
        if isinstance(fx_b, list):
            result.extend(fx_b)
        if isinstance(fx_c, list):
            result.extend(fx_c)
        return (result,)


class MovisBuildShader:
    """Build a MOVIS_SHADER config and wire it into any Add*Track node's optional shader input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shader_path": ("STRING", {"default": "", "placeholder": "GLSL shader 文件路径（可空）"}),
                "shader_code": ("STRING", {"default": "", "multiline": True, "placeholder": "可直接粘贴 GLSL 代码；非空时优先于 shader_path"}),
                "backend": (GPU_SHADER_BACKENDS, {"default": "libplacebo_vulkan"}),
                "codec": (GPU_SHADER_CODECS, {"default": "libx264"}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
                "output_file_prefix": ("STRING", {"default": "movis_clip_shader_"}),
            },
            "optional": {
                "chain": ("MOVIS_SHADER", {"tooltip": "可选：将本shader追加到已有 MOVIS_SHADER 列表末尾"}),
            },
        }

    RETURN_TYPES = ("MOVIS_SHADER",)
    RETURN_NAMES = ("shader",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "build_shader"

    def build_shader(self, shader_path, shader_code, backend, codec, keep_audio, enabled, output_file_prefix, chain=None):
        entry = _normalize_clip_shader_entry({
            "shader_path": shader_path,
            "shader_code": shader_code,
            "backend": backend,
            "codec": codec,
            "keep_audio": bool(keep_audio),
            "enabled": bool(enabled),
            "output_file_prefix": output_file_prefix,
        })
        result = list(chain) if isinstance(chain, list) else []
        if entry is not None:
            result.append(entry)
        return (result,)


# ---------------------------------------------------------------------------
# MOVIS_AUDIO builder nodes
# ---------------------------------------------------------------------------
# Design principle (mirrors MOVIS_FX pattern):
#   MovisBuildAudio  → build one MOVIS_AUDIO entry, wire into any track node's
#                       optional `audio` input.  Supports `chain` to accumulate
#                       multiple entries into one list.
#   MovisChainAudio  → merge up to 3 MOVIS_AUDIO lists (like MovisChainFX).
#
# This lets you attach dialogue + SFX + music stems directly on a single clip
# without threading them through separate AddAudioTrack nodes.
# ---------------------------------------------------------------------------

class MovisBuildAudio:
    """Build a MOVIS_AUDIO entry and wire it into any Add*Track node's optional
    ``audio`` input.

    Multiple entries can be accumulated via the ``chain`` input (mirrors
    MovisBuildFX chaining), so a single clip can carry arbitrary audio layers.

    ``offset_from_clip`` lets you delay the audio relative to the clip start
    (e.g. 0.5 s into the clip), useful for staggered SFX.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_level_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 40.0, "tooltip": "音轨增益 dB"}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "源文件截取起点（秒）"}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "时长；0 = 自动读取源文件时长"}),
                "offset_from_clip": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "tooltip": "相对于所绑定视频片段起点的偏移（秒）；0 = 与视频同步起始"}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": "", "placeholder": "音频路径；为空则用 audio 输入"}),
                "audio": ("AUDIO", {"tooltip": "ComfyUI 原生 AUDIO 输入；优先于 audio_path"}),
                "chain": ("MOVIS_AUDIO", {"tooltip": "将本条目追加到已有 MOVIS_AUDIO 列表末尾，实现多层叠加"}),
            },
        }

    RETURN_TYPES = ("MOVIS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "build_audio"

    def build_audio(self, audio_level_db, source_start, duration, offset_from_clip,
                    audio_path="", audio=None, chain=None):
        path = _resolve_audio_input(audio_path or "", audio=audio, prefix="movis_build_audio")
        if not path or not os.path.exists(path):
            return (list(chain) if isinstance(chain, list) else [],)
        dur = _safe_float(duration, 0.0, min_value=0.0)
        if dur <= 0:
            dur = _audio_duration(path)
        entry = {
            "path": path,
            "duration": max(0.01, dur),
            "source_start": _safe_float(source_start, 0.0, min_value=0.0),
            "audio_level_db": _safe_float(audio_level_db, 0.0, min_value=-60.0, max_value=40.0),
            "offset": _safe_float(offset_from_clip, 0.0),
        }
        result = list(chain) if isinstance(chain, list) else []
        result.append(entry)
        return (result,)


class MovisChainAudio:
    """Merge up to three MOVIS_AUDIO lists into one.

    Useful when you have pre-built audio groups (e.g. dialogue list + SFX list)
    that you want combined before wiring into a track node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("MOVIS_AUDIO",),
            },
            "optional": {
                "audio_b": ("MOVIS_AUDIO",),
                "audio_c": ("MOVIS_AUDIO",),
            },
        }

    RETURN_TYPES = ("MOVIS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "chain_audio"

    def chain_audio(self, audio_a, audio_b=None, audio_c=None):
        result = list(audio_a) if isinstance(audio_a, list) else []
        if isinstance(audio_b, list):
            result.extend(audio_b)
        if isinstance(audio_c, list):
            result.extend(audio_c)
        return (result,)


# ---------------------------------------------------------------------------
# MovisBatchAddVideoTracks — add N videos to the timeline in one node
# ---------------------------------------------------------------------------
# Eliminates the need to chain N AddVideoTrack nodes for simple montage work.
# Each path gets the same shared settings; per-clip override is via JSON.
# ---------------------------------------------------------------------------

class MovisBatchAddVideoTracks:
    """Add multiple video clips to a timeline in one node.

    ``video_paths`` is a newline- or comma-separated list of file paths.
    All clips share the same visual/transition settings; use
    ``per_clip_override_json`` for per-clip overrides::

        [
          {"audio_level_db": -6, "fade_in": 0.5},
          {},
          {"duration": 3.0}
        ]

    An empty dict ``{}`` means "use shared defaults for this clip".
    Shared ``fx``, ``shader``, and ``audio`` (MOVIS_AUDIO) are baked into
    every clip in the batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("MOVIS_TIMELINE",),
                "video_paths": ("STRING", {"default": "", "multiline": True,
                                           "placeholder": "每行一个视频路径，或逗号分隔",
                                           "tooltip": "多视频路径，换行或逗号分隔"}),
                "placement_mode": (["append", "absolute"], {"default": "append",
                                    "tooltip": "append: 顺序拼接；absolute: 全部从 start 开始（用于多图层叠加）"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0,
                                    "tooltip": "仅 absolute 模式有效"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "use_source_audio": ("BOOLEAN", {"default": True}),
                "audio_level_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 24.0}),
                "transition_in": (TRANSITION_TYPES, {"default": "none"}),
                "transition_in_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_out": (TRANSITION_TYPES, {"default": "none"}),
                "transition_out_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
            },
            "optional": {
                "vhs_filenames": ("VHS_FILENAMES",),
                "fx": ("MOVIS_FX", {"tooltip": "可选：对批次内所有片段应用同一 FX"}),
                "shader": ("MOVIS_SHADER", {"tooltip": "可选：对批次内所有片段应用同一 Shader"}),
                "audio": ("MOVIS_AUDIO", {"tooltip": "可选：对批次内所有片段绑定同一音频组"}),
                "per_clip_override_json": ("STRING", {"default": "",
                                           "multiline": True,
                                           "placeholder": "[{}, {\"duration\": 3.0}, {}]",
                                           "tooltip": "JSON 数组，每项对应一个片段的覆盖参数"}),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE", "INT")
    RETURN_NAMES = ("timeline", "clip_count")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "batch_add"

    def batch_add(self, timeline, video_paths, placement_mode, start, fade_in, fade_out,
                  use_source_audio, audio_level_db, transition_in, transition_in_duration,
                  transition_out, transition_out_duration, transition_easing,
                  vhs_filenames=None, fx=None, shader=None, audio=None,
                  per_clip_override_json=""):
        t = _clone_timeline(timeline)

        # Resolve video path list
        paths: list[str] = []
        if vhs_filenames and isinstance(vhs_filenames, (list, tuple)) and len(vhs_filenames) > 0:
            first = vhs_filenames[0]
            if isinstance(first, (list, tuple)):
                paths = [str(p) for p in first if p]
            else:
                paths = [str(p) for p in vhs_filenames if p]
        if not paths:
            raw = str(video_paths or "").replace(",", "\n")
            paths = [p.strip() for p in raw.splitlines() if p.strip()]

        # Parse per-clip overrides
        overrides: list[dict] = []
        ov_raw = str(per_clip_override_json or "").strip()
        if ov_raw:
            try:
                import json as _json
                parsed = _json.loads(ov_raw)
                if isinstance(parsed, list):
                    overrides = [x if isinstance(x, dict) else {} for x in parsed]
            except Exception:
                pass

        base_transition_kw = dict(
            transition_in=transition_in,
            duration_in=transition_in_duration,
            transition_out=transition_out,
            duration_out=transition_out_duration,
            easing=transition_easing,
            softness=0.5,
            custom_curve="",
        )

        for idx, raw_path in enumerate(paths):
            try:
                path = _resolve_video_input(raw_path)
            except Exception:
                continue

            ov = overrides[idx] if idx < len(overrides) else {}
            clip_duration = _safe_float(ov.get("duration", 0.0), 0.0, min_value=0.0)
            if clip_duration <= 0:
                clip_duration = _video_duration(path)

            if placement_mode == "append":
                clip_start = _timeline_content_duration(t)
            else:
                clip_start = _safe_float(start, 0.0, min_value=0.0)

            clip_start += _safe_float(ov.get("start_offset", 0.0), 0.0)

            clip = {
                "path": path,
                "is_image": False,
                "start": clip_start,
                "duration": max(0.01, clip_duration),
                "source_start": _safe_float(ov.get("source_start", 0.0), 0.0, min_value=0.0),
                "fade_in": _safe_float(ov.get("fade_in", fade_in), 0.0, min_value=0.0),
                "fade_out": _safe_float(ov.get("fade_out", fade_out), 0.0, min_value=0.0),
                "use_source_audio": bool(ov.get("use_source_audio", use_source_audio)),
                "audio_level_db": _safe_float(ov.get("audio_level_db", audio_level_db),
                                               0.0, min_value=-60.0, max_value=24.0),
                "position_x": _safe_float(ov.get("position_x", 0.5), 0.5),
                "position_y": _safe_float(ov.get("position_y", 0.5), 0.5),
                "scale_x": _safe_float(ov.get("scale_x", 1.0), 1.0, min_value=0.01, max_value=20.0),
                "scale_y": _safe_float(ov.get("scale_y", 1.0), 1.0, min_value=0.01, max_value=20.0),
                "rotation": _safe_float(ov.get("rotation", 0.0), 0.0),
                "opacity": _safe_float(ov.get("opacity", 1.0), 1.0, min_value=0.0, max_value=1.0),
                "transition_in": _make_transition(
                    ov.get("transition_in", base_transition_kw["transition_in"]),
                    ov.get("transition_in_duration", base_transition_kw["duration_in"]),
                    base_transition_kw["easing"],
                    base_transition_kw["softness"],
                    base_transition_kw["custom_curve"],
                ),
                "transition_out": _make_transition(
                    ov.get("transition_out", base_transition_kw["transition_out"]),
                    ov.get("transition_out_duration", base_transition_kw["duration_out"]),
                    base_transition_kw["easing"],
                    base_transition_kw["softness"],
                    base_transition_kw["custom_curve"],
                ),
            }
            _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
            _attach_audio_to_clip(t, clip_start, audio)
            t["video_tracks"].append(clip)

        return (t, len(paths))


# ---------------------------------------------------------------------------
# MovisQuickBuildTimeline — create + populate timeline in one node
# ---------------------------------------------------------------------------
# Convenience node for the common case: N videos + optional N audios + BGM.
# Replaces the CreateTimeline → AddVideoTrack×N → AddAudioTrack×N → SetBGM chain.
# ---------------------------------------------------------------------------

class MovisQuickBuildTimeline:
    """Create a timeline and populate it with videos + audio tracks in one node.

    ``video_paths`` and ``audio_paths`` are newline-separated path lists.
    Each audio entry is matched to the video at the same index
    (index 0 audio → index 0 video clip, etc.).
    Extra audio paths beyond the video count are ignored.

    Use the full node chain (CreateTimeline + Add*Track×N) for complex scenarios.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "bg_color": ("STRING", {"default": "#000000"}),
                "video_paths": ("STRING", {"default": "", "multiline": True,
                                           "placeholder": "每行一个视频路径",
                                           "tooltip": "多视频路径，换行分隔，顺序拼接"}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "use_source_audio": ("BOOLEAN", {"default": True,
                                                  "tooltip": "是否使用视频自带音轨"}),
                "transition_in": (TRANSITION_TYPES, {"default": "none"}),
                "transition_in_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_out": (TRANSITION_TYPES, {"default": "none"}),
                "transition_out_duration": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0}),
                "transition_easing": (TRANSITION_EASINGS, {"default": "ease_in_out"}),
                "bgm_level_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 40.0}),
                "bgm_trim_to_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio_paths": ("STRING", {"default": "", "multiline": True,
                                           "placeholder": "每行一个音频路径；数量与视频对应",
                                           "tooltip": "逐条与视频对齐；第N条音频配第N个视频片段起点"}),
                "bgm_path": ("STRING", {"default": "",
                                        "placeholder": "背景音乐路径（可留空）"}),
                "audio_bgm": ("AUDIO", {"tooltip": "ComfyUI 原生 AUDIO 输入作为 BGM"}),
                "fx": ("MOVIS_FX",),
                "shader": ("MOVIS_SHADER",),
            },
        }

    RETURN_TYPES = ("MOVIS_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/movis"
    FUNCTION = "quick_build"

    def quick_build(self, width, height, fps, bg_color, video_paths,
                    fade_in, fade_out, use_source_audio,
                    transition_in, transition_in_duration,
                    transition_out, transition_out_duration, transition_easing,
                    bgm_level_db, bgm_trim_to_video,
                    audio_paths="", bgm_path="", audio_bgm=None,
                    fx=None, shader=None):
        safe_w = int(_safe_float(width, 1080.0, min_value=64.0, max_value=8192.0))
        safe_h = int(_safe_float(height, 1920.0, min_value=64.0, max_value=8192.0))
        safe_fps = _safe_float(fps, 30.0, min_value=1.0, max_value=240.0)
        t = _new_timeline(safe_w, safe_h, safe_fps, str(bg_color or "#000000"))

        video_list = [p.strip() for p in str(video_paths or "").replace(",", "\n").splitlines() if p.strip()]
        audio_list = [p.strip() for p in str(audio_paths or "").replace(",", "\n").splitlines() if p.strip()]

        for idx, raw_vpath in enumerate(video_list):
            try:
                vpath = _resolve_video_input(raw_vpath)
            except Exception:
                continue
            dur = _video_duration(vpath)
            clip_start = _timeline_content_duration(t)
            clip = {
                "path": vpath,
                "is_image": False,
                "start": clip_start,
                "duration": max(0.01, dur),
                "source_start": 0.0,
                "fade_in": _safe_float(fade_in, 0.0, min_value=0.0),
                "fade_out": _safe_float(fade_out, 0.0, min_value=0.0),
                "use_source_audio": bool(use_source_audio),
                "audio_level_db": 0.0,
                "position_x": 0.5,
                "position_y": 0.5,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "rotation": 0.0,
                "opacity": 1.0,
                "transition_in": _make_transition(transition_in, transition_in_duration,
                                                   transition_easing, 0.5, ""),
                "transition_out": _make_transition(transition_out, transition_out_duration,
                                                    transition_easing, 0.5, ""),
            }
            _attach_fx_shader_to_clip(clip, fx=fx, shader=shader)
            t["video_tracks"].append(clip)

            # Pair audio by index
            if idx < len(audio_list):
                try:
                    apath = resolve_media_path(audio_list[idx], must_exist=True)
                    adur = _audio_duration(apath)
                    t["audio_tracks"].append({
                        "path": apath,
                        "start": clip_start,
                        "duration": max(0.01, adur),
                        "source_start": 0.0,
                        "audio_level_db": 0.0,
                    })
                except Exception:
                    pass

        # BGM
        bgm_resolved = _resolve_audio_input(bgm_path or "", audio=audio_bgm, prefix="movis_quick_bgm")
        if bgm_resolved and os.path.exists(bgm_resolved):
            t["bgm"] = {
                "path": bgm_resolved,
                "audio_level_db": _safe_float(bgm_level_db, -12.0, min_value=-60.0, max_value=40.0),
                "source_start": 0.0,
                "duration": _audio_duration(bgm_resolved),
                "trim_to_video_length": bool(bgm_trim_to_video),
                "auto_duck_bgm": False,
            }

        return (t,)


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
