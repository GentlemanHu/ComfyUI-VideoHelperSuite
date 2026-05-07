import datetime
import gc
import json
import math
import os
import shutil
import subprocess
import ast
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .utils import ffmpeg_path

try:
    import stable_whisper as whisper
except Exception:
    whisper = None


def _strip_quotes(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def resolve_media_path(path_like: str, must_exist: bool = True) -> str:
    """跨平台路径解析：支持绝对/相对/带引号路径。"""
    if not path_like:
        if must_exist:
            raise ValueError("path is empty")
        return ""

    raw = _strip_quotes(path_like)
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                Path.cwd() / p,
                Path(folder_paths.get_input_directory()) / p,
                Path(folder_paths.get_output_directory()) / p,
                Path(folder_paths.get_temp_directory()) / p,
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if not must_exist or resolved.exists():
            return str(resolved)

    if must_exist:
        raise FileNotFoundError(f"无法解析有效路径: {path_like}")
    return str((Path.cwd() / p).resolve())


class GentleCaption:
    """稳定字幕引擎（跨平台，避免硬编码 ffmpeg 路径）。"""

    def __init__(self) -> None:
        self.model = None
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "captioned")
        os.makedirs(self.output_dir, exist_ok=True)
        self.ffmpeg_exe = self._resolve_ffmpeg_exe()
        self.ffprobe_exe = self._resolve_ffprobe_exe(self.ffmpeg_exe)

    def _resolve_ffmpeg_exe(self) -> str:
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            return ffmpeg_path
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        raise FileNotFoundError("未找到 ffmpeg，可通过 PATH 或 VHS_FORCE_FFMPEG_PATH 配置")

    def _resolve_ffprobe_exe(self, ffmpeg_executable: str) -> str:
        ffmpeg_file = Path(ffmpeg_executable)
        probe_name = "ffprobe.exe" if ffmpeg_file.suffix.lower() == ".exe" else "ffprobe"
        sibling = ffmpeg_file.with_name(probe_name)
        if sibling.exists():
            return str(sibling)
        system_probe = shutil.which("ffprobe")
        if system_probe:
            return system_probe
        raise FileNotFoundError("未找到 ffprobe，可通过 PATH 安装")

    def _ensure_model(self):
        if self.model is not None:
            return
        if whisper is None:
            raise RuntimeError("stable_whisper 不可用，请安装 requirements.txt 中对应依赖")
        self.model = whisper.load_model("base")

    def release_model(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _parse_float(self, raw: str, field_name: str) -> float:
        text = (raw or "").strip()
        if not text:
            raise RuntimeError(f"ffprobe 未返回 {field_name}")
        return float(text)

    def get_media_info(self, filename: str, kind: str = "video") -> dict[str, Any]:
        src = resolve_media_path(filename, must_exist=True)
        if kind == "video":
            cmd = [
                self.ffprobe_exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate:format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                src,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe 视频信息失败: {result.stderr}")
            lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
            if len(lines) < 4:
                raise RuntimeError(f"ffprobe 视频信息输出异常: {result.stdout}")
            width = int(float(lines[0]))
            height = int(float(lines[1]))
            fps_raw = lines[2]
            if "/" in fps_raw:
                n, d = fps_raw.split("/", 1)
                fps = float(n) / max(float(d), 1.0)
            else:
                fps = float(fps_raw)
            duration = self._parse_float(lines[3], "duration")
            frames = max(1, int(round(duration * max(fps, 1e-6))))
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "frames": frames,
            }

        if kind == "audio":
            cmd = [
                self.ffprobe_exe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                src,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe 音频信息失败: {result.stderr}")
            duration = self._parse_float(result.stdout, "duration")
            return {"duration": duration}

        raise ValueError(f"不支持的媒体类型: {kind}")

    def _ass_filter_path(self, ass_path: str) -> str:
        p = str(Path(ass_path).resolve()).replace("\\", "/")
        p = p.replace(":", "\\:")
        p = p.replace("'", r"\'")
        return p

    def _parse_caption_params(self, caption_json_param: str) -> dict:
        txt = (caption_json_param or "").strip()
        if not txt:
            return {}

        if not txt.startswith("{"):
            txt = "{" + txt
        if not txt.endswith("}"):
            txt = txt + "}"

        normalized = txt.replace("'", '"')
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as e:
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError(f"caption_json_param 不是合法 JSON/Python 字典: {e}")

    def srt_create(self, word_dict: dict, audio_file: str, path: str | None = None) -> str:
        self._ensure_model()
        audio_file = resolve_media_path(audio_file, must_exist=True)
        path = path or self.output_dir
        os.makedirs(path, exist_ok=True)

        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        srt_filename = os.path.join(path, f"{stamp}.srt")
        ass_filename = os.path.join(path, f"{stamp}.ass")

        transcribe = None
        try:
            transcribe = self.model.transcribe(
                audio_file,
                regroup=True,
                fp16=torch.cuda.is_available(),
            )
            transcribe.split_by_gap(0.5).split_by_length(38).merge_by_gap(0.15, max_words=2)
            transcribe.to_srt_vtt(str(Path(srt_filename).resolve()), word_level=True)
            transcribe.to_ass(str(Path(ass_filename).resolve()), **word_dict)
        finally:
            transcribe = None
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return str(Path(ass_filename).resolve())

    def make_video(
        self,
        bg_video_path: str,
        bg_audio_path: str,
        output_filename: str,
        extra_para: dict,
        is_vertical: bool = True,
    ):
        video_path = resolve_media_path(bg_video_path, must_exist=True)
        audio_path = resolve_media_path(bg_audio_path, must_exist=True) if bg_audio_path else video_path

        video_info = self.get_media_info(video_path, kind="video")
        audio_info = self.get_media_info(audio_path, kind="audio")
        audio_duration = max(0.01, float(audio_info["duration"]))
        frames = max(1, int(round(video_info["fps"] * audio_duration)))

        ass_path = self.srt_create(extra_para, audio_path, path=self.output_dir)
        escaped_ass = self._ass_filter_path(ass_path)

        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        real_filename = f"{output_filename}_{stamp}.mp4"
        output_path = os.path.join(folder_paths.get_output_directory(), real_filename)

        if is_vertical:
            video_filter = f"crop=ih/16*9:ih,scale=1080:1920:flags=lanczos,ass='{escaped_ass}'"
        else:
            video_filter = f"scale=1920:1080:flags=lanczos,ass='{escaped_ass}'"

        args = [
            self.ffmpeg_exe,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-t",
            f"{audio_duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-filter:v",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            output_path,
        ]
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 字幕渲染失败:\n{result.stderr}")

        return real_filename, output_path, frames

    # 兼容旧接口
    def makeVideo(self, bg_video_path: str, bg_audio_path: str, output_filename: str, extra_para: dict):
        real_filename, output_path, _ = self.make_video(
            bg_video_path=bg_video_path,
            bg_audio_path=bg_audio_path,
            output_filename=output_filename,
            extra_para=extra_para,
            is_vertical=True,
        )
        return real_filename, output_path

    def parse_caption_params(self, caption_json_param: str) -> dict:
        return self._parse_caption_params(caption_json_param)

    def estimate_frames(self, video_path: str, audio_path: str | None = None) -> int:
        info = self.get_media_info(video_path, kind="video")
        if audio_path:
            audio_duration = self.get_media_info(audio_path, kind="audio")["duration"]
            return max(1, int(math.floor(info["fps"] * audio_duration + 0.5)))
        return int(info["frames"])
