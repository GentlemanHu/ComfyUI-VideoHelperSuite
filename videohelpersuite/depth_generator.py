import os
import platform
import subprocess
import sys
from datetime import datetime
from PIL import Image
import numpy as np
import re
import hashlib
import threading
import requests
from pathlib import Path
from urllib.parse import urlparse
import shutil
import torch
import cv2

import folder_paths
from comfy.utils import ProgressBar


def _detect_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available for OpenGL/EGL rendering."""
    # Quick check: nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and "NVIDIA" in r.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: check if CUDA is available via torch (already loaded by ComfyUI)
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        pass
    return False


def _find_nvidia_egl_icd() -> str | None:
    """Find the NVIDIA EGL vendor ICD JSON on the system."""
    candidates = [
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "/usr/share/egl/egl_external_platform.d/10_nvidia.json",
        "/etc/glvnd/egl_vendor.d/10_nvidia.json",
        "/usr/lib/x86_64-linux-gnu/egl_vendor.d/10_nvidia.json",
        "/usr/share/glvnd/egl_vendor.d/10_nvidia_wayland.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Glob search
    import glob
    for pattern in (
        "/usr/share/glvnd/egl_vendor.d/*nvidia*.json",
        "/etc/glvnd/egl_vendor.d/*nvidia*.json",
    ):
        found = glob.glob(pattern)
        if found:
            return found[0]
    return None


def _build_depthflow_env() -> dict:
    """
    Build the environment dict for the DepthFlow subprocess.

    On Linux headless with an NVIDIA GPU, forces EGL through the NVIDIA
    driver so that OpenGL rendering uses the GPU (not llvmpipe).
    """
    env = os.environ.copy()
    env["SHADERFLOW_BACKEND"] = "headless"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if platform.system() == "Windows":
        return env

    # ---- Linux ----
    env["PYOPENGL_PLATFORM"] = "egl"

    has_nvidia = _detect_nvidia_gpu()
    if has_nvidia:
        # Force NVIDIA EGL so moderngl/ShaderFlow picks up the real GPU
        egl_icd = _find_nvidia_egl_icd()
        if egl_icd:
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_icd
            print(f"[DepthFlow] NVIDIA EGL ICD found: {egl_icd}")

        # Tell the driver to render offscreen on NVIDIA even without X/Wayland
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        # Do NOT set LIBGL_ALWAYS_SOFTWARE when we have a real GPU
        env.pop("LIBGL_ALWAYS_SOFTWARE", None)
        env.pop("MESA_GL_VERSION_OVERRIDE", None)
        print("[DepthFlow] NVIDIA GPU detected → forcing EGL+NVIDIA for OpenGL")
    else:
        # Software fallback
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")
        env.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
        print("[DepthFlow] No NVIDIA GPU → using Mesa software OpenGL")

    return env


def _has_nvidia_egl_runtime() -> bool:
    """
    Check if NVIDIA EGL rendering is *actually* functional at runtime.
    Returns True only if libEGL_nvidia.so.0 can be loaded (meaning GPU
    OpenGL will work). CUDA availability alone is NOT enough.
    """
    if platform.system() == "Windows":
        return True  # Windows always has GPU OpenGL when driver is installed
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libEGL_nvidia.so.0")
        return True
    except (OSError, Exception):
        return False


def _ffmpeg_upscale_video(input_path: str, output_path: str, target_w: int,
                          target_h: int, codec: str = "h264") -> str:
    """
    Upscale a video to target resolution using ffmpeg.
    Tries NVENC encoder first (GPU accelerated), falls back to CPU libx264.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    # Map codec names to ffmpeg encoder names
    nvenc_map = {
        "h264": "h264_nvenc",
        "h264-nvenc": "h264_nvenc",
        "h265": "hevc_nvenc",
        "h265-nvenc": "hevc_nvenc",
    }
    cpu_map = {
        "h264": "libx264",
        "h264-nvenc": "libx264",
        "h265": "libx265",
        "h265-nvenc": "libx265",
        "av1-svt": "libsvtav1",
    }

    scale_filter = f"scale={target_w}:{target_h}:flags=lanczos"

    # Try NVENC first (GPU upscale+encode)
    nvenc = nvenc_map.get(codec)
    if nvenc:
        cmd = [
            ffmpeg_bin, "-y", "-v", "error",
            "-i", input_path,
            "-vf", scale_filter,
            "-c:v", nvenc,
            "-preset", "p4", "-rc", "vbr", "-cq", "18",
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[DepthFlow] ✓ NVENC upscale to {target_w}x{target_h} succeeded")
            return output_path

    # CPU fallback
    cpu_enc = cpu_map.get(codec, "libx264")
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", input_path,
        "-vf", scale_filter,
        "-c:v", cpu_enc,
        "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg upscale failed: {r.stderr.strip()}")
    print(f"[DepthFlow] ✓ CPU upscale to {target_w}x{target_h} succeeded")
    return output_path


def _find_depthflow_executable() -> str:
    """
    Cross-platform DepthFlow executable discovery.
    
    Search order:
      1. DEPTHFLOW_PATH env var (explicit user override)
      2. .venv_depthflow inside the VHS custom node directory (isolated venv)
      3. System PATH (via shutil.which)
      4. Common install locations per platform
    
    Returns the absolute path to the depthflow executable.
    Raises FileNotFoundError with helpful instructions if not found.
    """
    is_win = platform.system() == "Windows"
    exe_name = "depthflow.exe" if is_win else "depthflow"

    # --- 1. Explicit env var ---
    env_path = os.environ.get("DEPTHFLOW_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        # If user points to a directory, look for the exe inside
        if p.is_dir():
            for sub in (
                p / "venv" / ("Scripts" if is_win else "bin") / exe_name,
                p / ".venv" / ("Scripts" if is_win else "bin") / exe_name,
                p / ("Scripts" if is_win else "bin") / exe_name,
                p / exe_name,
            ):
                if sub.is_file():
                    return str(sub.resolve())
        elif p.is_file():
            return str(p.resolve())

    # --- 2. .venv_depthflow inside VHS node directory ---
    vhs_dir = Path(__file__).resolve().parent.parent  # ComfyUI-VideoHelperSuite root
    venv_dir = vhs_dir / ".venv_depthflow"
    if venv_dir.is_dir():
        scripts_dir = venv_dir / ("Scripts" if is_win else "bin")
        candidate = scripts_dir / exe_name
        if candidate.is_file():
            return str(candidate.resolve())

    # --- 3. System PATH ---
    which = shutil.which("depthflow")
    if which:
        return str(Path(which).resolve())

    # --- 4. Common install locations ---
    common_locations = []
    if is_win:
        common_locations = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DepthFlow" / "depthflow.exe",
            Path.home() / "DepthFlow" / "venv" / "Scripts" / "depthflow.exe",
        ]
        # Also check drives for D:\Play\tool\DepthFlow style installs
        for drive in ("C:", "D:", "E:"):
            common_locations.append(Path(drive) / "Play" / "tool" / "DepthFlow" / "venv" / "Scripts" / "depthflow.exe")
    else:
        common_locations = [
            Path.home() / ".local" / "bin" / "depthflow",
            Path("/usr/local/bin/depthflow"),
            Path("/usr/bin/depthflow"),
        ]

    for loc in common_locations:
        try:
            if loc.is_file():
                return str(loc.resolve())
        except (OSError, ValueError):
            continue

    # --- Not found: build helpful message ---
    venv_hint = str(vhs_dir / ".venv_depthflow")
    raise FileNotFoundError(
        f"DepthFlow executable not found.\n"
        f"Search locations tried:\n"
        f"  1. DEPTHFLOW_PATH env var: {env_path or '(not set)'}\n"
        f"  2. VHS venv: {venv_hint}\n"
        f"  3. System PATH\n"
        f"  4. Common install locations\n\n"
        f"To fix, do ONE of the following:\n"
        f"  - Create an isolated venv at {venv_hint}:\n"
        f"      python3.11 -m venv {venv_hint}\n"
        f"      {venv_hint}/{'Scripts' if is_win else 'bin'}/pip install depthflow\n"
        f"  - Set the DEPTHFLOW_PATH environment variable to the install directory or executable\n"
        f"  - Install depthflow globally so it appears on PATH\n"
    )

# Video cache directory
VIDEO_CACHE_DIR = os.path.join(folder_paths.get_temp_directory(), "video_cache")
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

class VideoDownloadManager:
    """Manages async video downloads with caching"""
    
    def __init__(self):
        self.downloads = {}  # url -> {status, path, progress}
        self.lock = threading.Lock()
    
    def get_cache_path(self, url):
        """Generate cache path from URL hash"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = os.path.splitext(urlparse(url).path)[1] or '.mp4'
        return os.path.join(VIDEO_CACHE_DIR, f"{url_hash}{ext}")
    
    def is_cached(self, url):
        """Check if video is already cached"""
        cache_path = self.get_cache_path(url)
        return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
    
    def get_status(self, url):
        """Get download status"""
        with self.lock:
            if url in self.downloads:
                return self.downloads[url]
            elif self.is_cached(url):
                return {"status": "completed", "path": self.get_cache_path(url), "progress": 100}
            return {"status": "not_started", "path": None, "progress": 0}
    
    def download_async(self, url, callback=None):
        """Start async download"""
        cache_path = self.get_cache_path(url)
        
        # Check if already cached
        if self.is_cached(url):
            print(f"[VideoCache] Using cached video: {cache_path}")
            if callback:
                callback(cache_path)
            return cache_path
        
        # Check if already downloading
        with self.lock:
            if url in self.downloads and self.downloads[url]["status"] == "downloading":
                print(f"[VideoCache] Already downloading: {url}")
                return self.downloads[url]["path"]
            
            self.downloads[url] = {"status": "downloading", "path": cache_path, "progress": 0}
        
        # Start download thread
        thread = threading.Thread(target=self._download_worker, args=(url, cache_path, callback))
        thread.daemon = True
        thread.start()
        
        return cache_path
    
    def _download_worker(self, url, cache_path, callback):
        """Worker thread for downloading"""
        try:
            print(f"[VideoCache] Downloading: {url}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            temp_path = cache_path + ".tmp"
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            with self.lock:
                                if url in self.downloads:
                                    self.downloads[url]["progress"] = progress
            
            # Move temp file to final location
            shutil.move(temp_path, cache_path)
            
            with self.lock:
                self.downloads[url] = {"status": "completed", "path": cache_path, "progress": 100}
            
            print(f"[VideoCache] Download completed: {cache_path}")
            
            if callback:
                callback(cache_path)
                
        except Exception as e:
            print(f"[VideoCache] Download failed: {e}")
            with self.lock:
                self.downloads[url] = {"status": "failed", "path": None, "progress": 0, "error": str(e)}
            
            # Clean up temp file
            temp_path = cache_path + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Global download manager
video_manager = VideoDownloadManager() 

class DepthFlowGenerator:
    """
    Professional DepthFlow video generator with cinematic camera movements
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "orientation": (["landscape", "portrait", "square"], {
                    "default": "landscape",
                    "tooltip": "屏幕方向 | landscape: 横屏(16:9) | portrait: 竖屏(9:16) | square: 方形(1:1)"
                }),
                "camera_movement": (["vertical", "horizontal", "zoom", "circle", "dolly", "orbital", "static"], {
                    "default": "vertical",
                    "tooltip": "运镜模式 | vertical: 上下移动 | horizontal: 左右移动 | zoom: 推拉镜头 | circle: 环绕运动 | dolly: 推轨变焦(电影感) | orbital: 轨道环绕 | static: 静止"
                }),
            },
            "optional": {
                # Resolution settings
                "resolution_mode": (["original", "custom", "1080p", "2k", "4k", "8k"], {
                    "default": "original",
                    "tooltip": "分辨率模式 | original: 保持原始分辨率 | custom: 自定义分辨率 | 1080p/2k/4k/8k: 预设分辨率(根据orientation自动调整)"
                }),
                "custom_width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "自定义宽度(仅在resolution_mode=custom时生效) | 范围: 64-8192 | 建议: 8的倍数"
                }),
                "custom_height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "自定义高度(仅在resolution_mode=custom时生效) | 范围: 64-8192 | 建议: 8的倍数"
                }),
                "crop_mode": (["center", "fit", "fill", "stretch"], {
                    "default": "fit",
                    "tooltip": "裁剪模式 | fit: 适应(保持比例,可能有黑边) | fill: 填充(裁剪以填满) | center: 中心裁剪 | stretch: 拉伸(可能变形)"
                }),
                
                # Video settings
                "duration": ("FLOAT", {
                    "default": 8.0, "min": 0, "max": 120.0, "step": 0.01,
                    "tooltip": "视频时长(秒) | 范围: 0.01-120秒 | 推荐: 6-12秒 | 注意: 时长越长渲染越慢"
                }),
                "fps": ("INT", {
                    "default": 60, "min": 1, "max": 120, "step": 1,
                    "tooltip": "帧率(每秒帧数) | 范围: 1-120 | 推荐: 30(快速) 60(流畅) | 注意: 高帧率渲染更慢"
                }),
                
                # Quality settings
                "quality": ("INT", {
                    "default": 100, "min": 1, "max": 100, "step": 1,
                    "tooltip": "视频质量 | 范围: 1-100 | 推荐: 80(预览) 90(标准) 100(最高) | 注意: 高质量文件更大"
                }),
                "ssaa": ("FLOAT", {
                    "default": 1.0, "min": 0, "max": 4.0, "step": 0.01,
                    "tooltip": "超采样抗锯齿 | 范围: 0.1-4.0 | 推荐: 0.5(快速) 1.0(标准) 2.0(高质量) | 注意: 值越高GPU负载越大(N²倍)"
                }),
                
                # Camera movement parameters
                "movement_intensity": ("FLOAT", {
                    "default": 1.0, "min": 0, "max": 4.0, "step": 0.01,
                    "tooltip": "运动强度 | 范围: 0-4 | 推荐: 0.3(微妙) 1.0(标准) 2.0(强烈) | 适用: 所有运镜模式"
                }),
                "movement_smooth": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "平滑运动(正弦波) | True: 平滑缓动 | False: 线性匀速 | 适用: vertical, horizontal, zoom, dolly"
                }),
                "movement_loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "循环运动 | True: 无缝循环 | False: 单次运动 | 适用: vertical, horizontal, zoom, dolly"
                }),
                "movement_reverse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "反向运动 | True: 反向播放 | False: 正向播放 | 适用: 所有运镜模式(除static)"
                }),
                "movement_phase": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "相位偏移 | 范围: 0-1 | 说明: 动画起始位置 | 0=起点 0.5=中点 1=终点 | 适用: vertical, horizontal, zoom, dolly, circle"
                }),
                
                # Depth settings
                "steady_depth": ("FLOAT", {
                    "default": 0.3, "min": -1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "稳定深度/焦点深度 | 范围: -1到2 | -1: 前景稳定 | 0-0.5: 中景稳定 | 1+: 后景稳定 | 适用: vertical/horizontal/circle(作为稳定深度), dolly/orbital(作为焦点深度)"
                }),
                "isometric": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "等距投影强度 | 范围: 0-1 | 0: 完全透视(强3D感) | 0.6: 平衡(推荐) | 1: 完全等距(平面感) | 适用: vertical, horizontal, zoom, circle"
                }),
                
                # Export settings
                "output_format": (["mp4", "mkv", "webm", "avi", "gif"], {
                    "default": "mp4",
                    "tooltip": "输出格式 | mp4: 最兼容 | mkv: 高质量容器 | webm: 网页优化 | avi: 传统格式 | gif: 动图(无需编码器)"
                }),
                "video_codec": (["h264", "h265", "h264-nvenc", "h265-nvenc", "av1-svt"], {
                    "default": "h264",
                    "tooltip": "视频编码器 | h264: CPU编码,稳定,支持任意分辨率(推荐) | h265: CPU编码,更好压缩 | h264-nvenc: GPU加速,快速但限制≤4K | h265-nvenc: GPU加速HEVC | av1-svt: 最佳压缩,最慢"
                }),
                
                # Depth estimator
                "depth_estimator": (["da2", "da1", "depthpro", "zoedepth", "marigold"], {
                    "default": "da2",
                    "tooltip": "深度估计模型 | da2: Depth Anything V2(推荐,快速) | da1: Depth Anything V1 | depthpro: Apple DepthPro(高质量) | zoedepth: ZoeDepth | marigold: Marigold(慢但准确)"
                }),
                
                # Depth extraction settings
                "output_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "输出视频帧 | True: 生成并返回所有视频帧 | False: 仅生成视频文件"
                }),
                "max_frames_export": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "最多导出帧数 | 0: 导出所有帧 | >0: 限制帧数(防止内存溢出) | 建议: 250-500帧"
                }),

                # CUDA Inpaint settings
                "cuda_enable_inpaint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "CUDA路径：修复大视差边缘拉伸/撕裂。用原始图像平滑替换陟岭区域。"
                }),
                "cuda_inpaint_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "CUDA路径：陟岭检测阈值。越低修复越多，但可能影响边缘细节。推荐0.3-1.0。"
                }),
                "cuda_inpaint_blur": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "CUDA路径：过渡融合半径。越大融合越平滑，0=硬边缘。推荐3-8。"
                }),
                "cuda_enable_aa": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "CUDA路径：抗锯齿滤波。建议开启。"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frames")
    OUTPUT_NODE = True
    FUNCTION = "run_depthflow"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    
    def extract_video_frames(self, video_path, max_frames=None, use_memory_efficient_mode=False, show_progress=True):
        """
        从视频文件中高效提取所有帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最多提取的帧数（None=无限制）
            use_memory_efficient_mode: 如果为True，分批加载以节省内存
        
        Returns:
            torch.Tensor: 形状 (N, H, W, 3) 的视频帧张量，值范围 0-1
        """
        print(f"[DepthFlow] Extracting frames from video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        try:
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[DepthFlow] Video info: {total_frames} frames @ {fps:.1f}fps, {frame_width}x{frame_height}")
            
            # 限制帧数（防止内存溢出）
            if max_frames is None:
                frames_to_load = total_frames
            else:
                frames_to_load = min(total_frames, max_frames)
            
            print(f"[DepthFlow] Loading {frames_to_load} frames...")
            
            # 创建进度条
            pbar = ProgressBar(frames_to_load) if show_progress else None
            
            # 预分配帧数组
            frames_list = []
            frame_index = 0
            
            while frame_index < frames_to_load:
                ret, frame = cap.read()
                if not ret:
                    print(f"[DepthFlow] Reached end of video at frame {frame_index}")
                    break
                
                # BGR 转 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为 float32，范围 0-1
                frame = frame.astype(np.float32) / 255.0
                
                frames_list.append(frame)
                frame_index += 1
                if pbar is not None:
                    pbar.update_absolute(frame_index, frames_to_load)
            
            if not frames_list:
                raise RuntimeError("No frames could be extracted from video")
            
            print(f"[DepthFlow] Successfully loaded {len(frames_list)} frames")
            
            # 转换为 torch tensor
            # 形状: (N, H, W, 3)
            frames_array = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_array).float()
            
            print(f"[DepthFlow] Frames tensor shape: {frames_tensor.shape}")
            print(f"[DepthFlow] Tensor memory usage: {frames_tensor.nbytes / (1024**2):.2f} MB")
            
            return frames_tensor
        
        finally:
            cap.release()
    
    def get_target_resolution(self, mode, original_width, original_height, custom_width, custom_height):
        """Calculate target resolution based on mode"""
        if mode == "original":
            return original_width, original_height
        elif mode == "custom":
            return custom_width, custom_height
        elif mode == "1080p":
            return 1920, 1080
        elif mode == "2k":
            return 2560, 1440
        elif mode == "4k":
            return 3840, 2160
        elif mode == "8k":
            return 7680, 4320
        return original_width, original_height
    
    def process_image_for_resolution(self, img, target_width, target_height, crop_mode):
        """Process image to target resolution with specified crop mode"""
        original_width, original_height = img.size
        
        if crop_mode == "stretch":
            # Simply resize without maintaining aspect ratio
            return img.resize((target_width, target_height), Image.LANCZOS)
        
        elif crop_mode == "fit":
            # Fit image inside target resolution (letterbox/pillarbox)
            img.thumbnail((target_width, target_height), Image.LANCZOS)
            # Create black background
            result = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            # Paste image centered
            offset = ((target_width - img.width) // 2, (target_height - img.height) // 2)
            result.paste(img, offset)
            return result
        
        elif crop_mode == "fill":
            # Fill target resolution (crop to fit)
            source_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if source_ratio > target_ratio:
                # Image is wider, scale by height
                new_height = target_height
                new_width = int(original_width * (target_height / original_height))
            else:
                # Image is taller, scale by width
                new_width = target_width
                new_height = int(original_height * (target_width / original_width))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Crop from center
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            return img.crop((left, top, left + target_width, top + target_height))
        
        elif crop_mode == "center":
            # Center crop without scaling
            if original_width < target_width or original_height < target_height:
                # If image is smaller, fit it
                return self.process_image_for_resolution(img, target_width, target_height, "fit")
            
            left = (original_width - target_width) // 2
            top = (original_height - target_height) // 2
            return img.crop((left, top, left + target_width, top + target_height))
        
        return img

    def run_depthflow(
        self,
        images,
        orientation="landscape",
        camera_movement="vertical",
        resolution_mode="original",
        custom_width=1920,
        custom_height=1080,
        crop_mode="fit",
        duration=8.0,
        fps=60,
        quality=100,
        ssaa=1.0,
        movement_intensity=1.0,
        movement_smooth=True,
        movement_loop=True,
        movement_reverse=False,
        movement_phase=0.0,
        steady_depth=0.3,
        isometric=0.6,
        output_format="mp4",
        video_codec="h264-nvenc",
        depth_estimator="da2",
        output_frames=True,
        max_frames_export=0,
        cuda_enable_inpaint=True,
        cuda_inpaint_threshold=0.5,
        cuda_inpaint_blur=5,
        cuda_enable_aa=True,
    ):
        """
        Professional DepthFlow video generation with cinematic camera movements
        """
        # Take the first image from the batch
        image = images[0]

        # Convert tensor to PIL Image
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        original_width, original_height = img.size
        print(f"[DepthFlow] Original image size: {original_width}x{original_height}")
        
        # Calculate target resolution based on orientation
        if resolution_mode == "original":
            target_width, target_height = original_width, original_height
        elif resolution_mode == "custom":
            target_width, target_height = custom_width, custom_height
        else:
            # Apply orientation to preset resolutions
            presets = {
                "1080p": (1920, 1080),
                "2k": (2560, 1440),
                "4k": (3840, 2160),
                "8k": (7680, 4320)
            }
            base_width, base_height = presets.get(resolution_mode, (1920, 1080))
            
            if orientation == "portrait":
                target_width, target_height = base_height, base_width
            elif orientation == "square":
                target_width = target_height = min(base_width, base_height)
            else:  # landscape
                target_width, target_height = base_width, base_height
        
        print(f"[DepthFlow] Target resolution: {target_width}x{target_height} ({orientation})")
        print(f"[DepthFlow] Crop mode: {crop_mode}")
        
        # Process image to target resolution
        processed_img = self.process_image_for_resolution(img, target_width, target_height, crop_mode)
        
        # Save processed image
        _datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"depthflow_input_{_datetime}.png"
        input_path = os.path.join(folder_paths.get_temp_directory(), input_filename)
        
        # Save with maximum quality
        processed_img.save(input_path, format='PNG', compress_level=0)
        print(f"[DepthFlow] Saved processed input: {input_path}")

        # Generate output filename
        output_filename = f"depthflow_{_datetime}_{target_width}x{target_height}.{output_format}"
        output_dir = os.path.join(folder_paths.get_output_directory(), "depthflow_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # ═══════════════════════════════════════════════════════════
        # ======  CUDA direct rendering (no subprocess/OpenGL)  =====
        # ═══════════════════════════════════════════════════════════
        cuda_done = False
        try:
            cuda_done = self._try_cuda_render(
                image_tensor=image,
                target_width=target_width,
                target_height=target_height,
                output_path=output_path,
                output_format=output_format,
                duration=duration,
                fps=fps,
                quality=quality,
                ssaa=ssaa,
                camera_movement=camera_movement,
                movement_intensity=movement_intensity,
                movement_smooth=movement_smooth,
                movement_loop=movement_loop,
                movement_reverse=movement_reverse,
                movement_phase=movement_phase,
                steady_depth=steady_depth,
                isometric=isometric,
                video_codec=video_codec,
                depth_estimator=depth_estimator,
                output_frames=output_frames,
                max_frames_export=max_frames_export,
                cuda_enable_inpaint=cuda_enable_inpaint,
                cuda_inpaint_threshold=cuda_inpaint_threshold,
                cuda_inpaint_blur=cuda_inpaint_blur,
                cuda_enable_aa=cuda_enable_aa,
            )
        except Exception as cuda_err:
            print(f"[DepthFlow] CUDA direct render failed: {cuda_err}")
            print(f"[DepthFlow] Falling back to subprocess OpenGL path...")
            cuda_done = False

        if cuda_done:
            final_path = os.path.abspath(output_path)
            print(f"[DepthFlow] ✓ CUDA direct render complete: {final_path}")
            # Clean up temp input
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
            except OSError:
                pass

            # Build frames tensor from captured frames (no video re-decode)
            frames_tensor = None
            captured = getattr(self, '_cuda_captured_frames', None)
            if output_frames and captured:
                print(f"[DepthFlow] Building tensor from {len(captured)} captured frames...")
                # Each frame is (H, W, 3) uint8 CPU tensor
                frames_tensor = torch.stack(captured).float() / 255.0
                print(f"[DepthFlow] ✓ Frames tensor: {frames_tensor.shape} "
                      f"({frames_tensor.nbytes / (1024**2):.0f} MB)")
                self._cuda_captured_frames = None  # free memory

            if frames_tensor is None:
                frames_tensor = torch.zeros(
                    (1, target_height, target_width, 3), dtype=torch.float32,
                )

            return {
                "ui": {
                    "video": [{
                        "filename": output_filename,
                        "subfolder": "depthflow_videos",
                        "type": "output",
                    }]
                },
                "result": (final_path, frames_tensor),
            }

        # ═══════════════════════════════════════════════════════════
        # ======  Subprocess OpenGL path (original)              =====
        # ═══════════════════════════════════════════════════════════

        # Use the full path to the depthflow executable
        depthflow_exe = _find_depthflow_executable()
        print(f"[DepthFlow] Using executable: {depthflow_exe}")
        
        # Build command with proper structure
        command = [
            depthflow_exe,
            "input", "-i", input_path,
            depth_estimator,  # Depth estimator
        ]
        
        # Add camera movement with parameters (if not static)
        if camera_movement != "static":
            command.append(camera_movement)
            command.extend(["--intensity", str(movement_intensity)])
            
            # Common parameters for most movements
            # vertical, horizontal, zoom, dolly support these
            if camera_movement in ["vertical", "horizontal", "zoom", "dolly"]:
                # Boolean flags - add appropriate flag for each state
                if movement_reverse:
                    command.append("-r")
                else:
                    command.append("-fw")
                
                if movement_smooth:
                    command.append("-s")
                else:
                    command.append("-ns")
                
                if movement_loop:
                    command.append("-l")
                else:
                    command.append("-nl")
                
                command.extend(["-p", str(movement_phase)])
            
            # circle supports reverse and phase
            elif camera_movement == "circle":
                if movement_reverse:
                    command.append("-r")
                else:
                    command.append("-fw")
                command.extend(["-p", str(movement_phase), str(movement_phase), str(movement_phase)])
            
            # orbital supports reverse only
            elif camera_movement == "orbital":
                if movement_reverse:
                    command.append("-r")
                else:
                    command.append("-fw")
            
            # Add steady_depth for movements that support it
            # vertical, horizontal, circle support -S
            if camera_movement in ["vertical", "horizontal", "circle"]:
                command.extend(["-S", str(steady_depth)])
            
            # Add isometric for movements that support it
            # vertical, horizontal, zoom, circle support -I
            if camera_movement in ["vertical", "horizontal", "zoom", "circle"]:
                command.extend(["-I", str(isometric)])
            
            # Add depth for movements that support it
            # dolly, orbital support -d
            if camera_movement in ["dolly", "orbital"]:
                command.extend(["-d", str(steady_depth)])  # Use steady_depth as focal depth
        
        # Add video codec (only if not gif)
        if output_format != "gif":
            command.append(video_codec)
        
        # ---- Detect if GPU OpenGL is available ----
        # If not (llvmpipe / cloud GPU without libEGL_nvidia), render at
        # reduced resolution then upscale with ffmpeg (NVENC if available).
        gpu_opengl = _has_nvidia_egl_runtime()
        render_width = target_width
        render_height = target_height
        needs_upscale = False

        if not gpu_opengl and platform.system() != "Windows":
            # Calculate a reasonable render resolution:
            # halve each dimension (4x fewer pixels → ~4x faster on CPU)
            # but never go below 640 on the short side
            MIN_DIM = 640
            scale = 0.5
            rw = max(MIN_DIM, int(target_width * scale))
            rh = max(MIN_DIM, int(target_height * scale))
            # keep even
            rw = rw - (rw % 2)
            rh = rh - (rh % 2)

            if rw < target_width or rh < target_height:
                render_width = rw
                render_height = rh
                needs_upscale = True
                # Also force SSAA to minimum to save more CPU time
                ssaa = min(ssaa, 0.5)
                print(f"[DepthFlow] ⚠ No NVIDIA OpenGL (llvmpipe CPU mode detected)")
                print(f"[DepthFlow] ⚠ Auto-optimizing: render at {render_width}x{render_height} "
                      f"(SSAA={ssaa}), then upscale to {target_width}x{target_height}")

        # Add main rendering parameters
        command.extend([
            "main",
            "-w", str(render_width),
            "-h", str(render_height),
            "-q", str(quality),
            "-t", str(duration),
            "-f", str(fps),
            "-s", str(ssaa),
            "-r",  # Render mode
            "-o", output_path,
            "--format", output_format,
        ])
        
        # Set environment variables for headless / cross-platform rendering
        env = _build_depthflow_env()
        
        final_path = os.path.abspath(output_path)
        
        # Run the depthflow command
        print(f"[DepthFlow] ==========================================")
        print(f"[DepthFlow] Starting video generation")
        print(f"[DepthFlow] Orientation: {orientation}")
        print(f"[DepthFlow] Target Resolution: {target_width}x{target_height}")
        if needs_upscale:
            print(f"[DepthFlow] Render Resolution: {render_width}x{render_height} (will upscale)")
        print(f"[DepthFlow] Camera Movement: {camera_movement}")
        print(f"[DepthFlow] Movement Intensity: {movement_intensity}x")
        print(f"[DepthFlow] Quality: {quality}%")
        print(f"[DepthFlow] Duration: {duration}s @ {fps}fps")
        print(f"[DepthFlow] SSAA: {ssaa}x")
        print(f"[DepthFlow] Codec: {video_codec}")
        print(f"[DepthFlow] Format: {output_format}")
        print(f"[DepthFlow] Depth Estimator: {depth_estimator}")
        print(f"[DepthFlow] GPU OpenGL: {'YES' if gpu_opengl else 'NO (llvmpipe CPU)'}")
        print(f"[DepthFlow] OpenGL Platform: {env.get('PYOPENGL_PLATFORM', 'default')}")
        print(f"[DepthFlow] EGL Vendor ICD: {env.get('__EGL_VENDOR_LIBRARY_FILENAMES', 'auto')}")
        print(f"[DepthFlow] Output: {final_path}")
        print(f"[DepthFlow] ==========================================")
        print(f"[DepthFlow] Command: {' '.join(command)}")
        print(f"[DepthFlow] ==========================================")
        
        # Create progress bar
        total_frames = int(duration * fps)
        pbar = ProgressBar(total_frames)
        
        try:
            # Run with real-time output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            frame_count = 0
            
            # Read output line by line in real-time
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Filter out Unicode error messages
                    if any(skip in line for skip in [
                        "UnicodeEncodeError", "gbk", "illegal multibyte",
                        "Logging error in Loguru Handler", "--- End of logging error ---"
                    ]):
                        continue
                    
                    # Print output
                    try:
                        print(f"[DepthFlow] {line}")
                    except UnicodeEncodeError:
                        print(f"[DepthFlow] {line.encode('ascii', 'replace').decode('ascii')}")
                    
                    # Extract frame progress
                    frame_match = re.search(r'(\d+)/(\d+)', line)
                    if frame_match:
                        current = int(frame_match.group(1))
                        total = int(frame_match.group(2))
                        pbar.update_absolute(current, total)
                    elif any(keyword in line.lower() for keyword in ['frame', 'rendering', 'encoding']):
                        frame_count += 1
                        if frame_count <= total_frames:
                            pbar.update_absolute(frame_count, total_frames)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            
            # Verify output file exists
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Output video was not created at: {final_path}")
            
            file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
            print(f"[DepthFlow] ==========================================")
            print(f"[DepthFlow] ✓ Video generation completed successfully!")
            print(f"[DepthFlow] ✓ Output: {final_path}")
            print(f"[DepthFlow] ✓ File size: {file_size_mb:.2f} MB")
            print(f"[DepthFlow] ==========================================")

            # ---- Upscale if we rendered at reduced resolution ----
            if needs_upscale and os.path.exists(final_path):
                print(f"[DepthFlow] Upscaling {render_width}x{render_height} → {target_width}x{target_height} ...")
                upscaled_path = final_path.replace(f".{output_format}", f"_upscaled.{output_format}")
                try:
                    _ffmpeg_upscale_video(final_path, upscaled_path,
                                          target_width, target_height, video_codec)
                    # Replace the original with the upscaled version
                    os.replace(upscaled_path, final_path)
                    new_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                    print(f"[DepthFlow] ✓ Upscale complete: {new_size_mb:.2f} MB")
                except Exception as upscale_err:
                    print(f"[DepthFlow] ⚠ Upscale failed ({upscale_err}), keeping {render_width}x{render_height} output")
                    # Clean up failed upscale file
                    if os.path.exists(upscaled_path):
                        try:
                            os.remove(upscaled_path)
                        except OSError:
                            pass
            
        except FileNotFoundError as e:
            raise
        except subprocess.CalledProcessError as e:
            error_msg = f"[DepthFlow] ✗ Error: Process failed with return code {e.returncode}"
            print(error_msg)
            
            # Provide helpful error messages
            if "h264_nvenc" in str(command) or "h265_nvenc" in str(command):
                print("[DepthFlow] ℹ️  NVENC encoder failed. This can happen when:")
                print("[DepthFlow]    - Resolution is too large for GPU encoder")
                print("[DepthFlow]    - GPU is busy with other tasks")
                print("[DepthFlow]    - Driver doesn't support NVENC")
                print("[DepthFlow] 💡 Try using 'h264' or 'h265' (CPU encoder) instead")
            
            if target_width > 4096 or target_height > 4096:
                print("[DepthFlow] ℹ️  Resolution is very large (>4K)")
                print("[DepthFlow] 💡 Try reducing resolution or using CPU encoder")
            
            raise RuntimeError(
                f"DepthFlow failed to generate video.\n"
                f"Return code: {e.returncode}\n"
                f"Try: Use 'h264' codec instead of 'h264-nvenc' for large resolutions"
            )
        finally:
            # Clean up temporary input file
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                    print(f"[DepthFlow] Cleaned up temporary input file")
            except:
                pass
        
        # 提取视频帧如果需要
        frames_tensor = None
        if output_frames:
            print(f"[DepthFlow] ==========================================")
            print(f"[DepthFlow] Extracting video frames...")
            try:
                # 确定最多提取的帧数
                max_frames = max_frames_export if max_frames_export > 0 else None
                
                # 从视频中提取所有帧
                frames_tensor = self.extract_video_frames(
                    final_path,
                    max_frames=max_frames,
                    use_memory_efficient_mode=False
                )
                print(f"[DepthFlow] ✓ Video frames extracted successfully")
                print(f"[DepthFlow] ==========================================")
            except Exception as e:
                print(f"[DepthFlow] ✗ Warning: Failed to extract frames: {e}")
                print(f"[DepthFlow] Continuing without frame extraction...")
                frames_tensor = None
        else:
            # 返回空张量以维持返回类型一致性
            frames_tensor = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)

        return {
            "ui": {
                "video": [{
                    "filename": output_filename,
                    "subfolder": "depthflow_videos",
                    "type": "output"
                }]
            },
            "result": (final_path, frames_tensor if frames_tensor is not None else torch.zeros((1, target_height, target_width, 3), dtype=torch.float32))
        }

    # ================================================================
    # CUDA direct rendering helpers
    # ================================================================

    def _finalize_output(self, final_path, output_filename,
                         target_width, target_height,
                         output_frames, max_frames_export,
                         show_progress=True):
        """Common output finalisation for both CUDA and subprocess paths."""
        frames_tensor = None
        if output_frames:
            print(f"[DepthFlow] Extracting video frames...")
            try:
                max_frames = max_frames_export if max_frames_export > 0 else None
                frames_tensor = self.extract_video_frames(
                    final_path, max_frames=max_frames,
                    use_memory_efficient_mode=False,
                    show_progress=show_progress,
                )
                print(f"[DepthFlow] ✓ Video frames extracted successfully")
            except Exception as e:
                print(f"[DepthFlow] ✗ Frame extraction failed: {e}")
                frames_tensor = None
        if frames_tensor is None:
            frames_tensor = torch.zeros(
                (1, target_height, target_width, 3), dtype=torch.float32,
            )
        return {
            "ui": {
                "video": [{
                    "filename": output_filename,
                    "subfolder": "depthflow_videos",
                    "type": "output",
                }]
            },
            "result": (final_path, frames_tensor),
        }

    def _try_cuda_render(
        self, *, image_tensor, target_width, target_height,
        output_path, output_format, duration, fps, quality, ssaa,
        camera_movement, movement_intensity, movement_smooth,
        movement_loop, movement_reverse, movement_phase,
        steady_depth, isometric, video_codec, depth_estimator,
        output_frames=True, max_frames_export=0,
        cuda_enable_inpaint=True,
        cuda_inpaint_threshold=0.5,
        cuda_inpaint_blur=5,
        cuda_enable_aa=True,
    ) -> bool:
        """Attempt to render directly on CUDA.  Returns True on success."""
        if not torch.cuda.is_available():
            return False

        # Try to import the CUDA renderer from the DepthFlow fork
        cuda_renderer = None

        # Build search paths: venv site-packages, local dev, env var
        vhs_dir = Path(__file__).resolve().parent.parent
        search_paths = []

        # 1. The .venv_depthflow's installed depthflow package (cloud / production)
        venv_sp = vhs_dir / ".venv_depthflow"
        if venv_sp.is_dir():
            for sp in (venv_sp / "lib").glob("python*/site-packages/depthflow"):
                search_paths.append(sp.parent.parent.parent.parent)  # back to venv root? No — we want the parent of depthflow/
                # Actually we want the dir containing depthflow/, which IS the site-packages dir
                # But our loop below looks for search_path / "depthflow" / "cuda_renderer.py"
                # So we need the site-packages dir itself
                search_paths.append(sp.parent)  # site-packages/
            # Windows venv layout
            win_sp = venv_sp / "Lib" / "site-packages" / "depthflow"
            if win_sp.is_dir():
                search_paths.append(win_sp.parent)

        # 2. Try to find via the depthflow module itself (if importable)
        try:
            import importlib
            df_mod = importlib.import_module("depthflow")
            df_init = Path(df_mod.__file__).resolve().parent
            search_paths.append(df_init.parent)
        except Exception:
            pass

        # 3. Local development paths
        search_paths.append(Path("D:/Play/DepthFlow"))
        search_paths.append(vhs_dir.parent.parent / "DepthFlow")

        # 4. DEPTHFLOW_PATH env var
        env_dp = os.environ.get("DEPTHFLOW_PATH", "").strip()
        if env_dp:
            search_paths.append(Path(env_dp))

        for search_path in search_paths:
            if search_path is None:
                continue
            mod_path = search_path / "depthflow" / "cuda_renderer.py"
            if mod_path.is_file():
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "depthflow.cuda_renderer", str(mod_path),
                )
                cuda_renderer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cuda_renderer)
                print(f"[DepthFlow] CUDA renderer found at: {mod_path}")
                break

        if cuda_renderer is None or not cuda_renderer.is_available():
            print("[DepthFlow] CUDA renderer module not found or CUDA unavailable")
            return False

        print("[DepthFlow] ═══════════════════════════════════════════")
        print("[DepthFlow]  🚀 CUDA Direct Rendering (no OpenGL)")
        print("[DepthFlow] ═══════════════════════════════════════════")

        # --- Prepare image (already a ComfyUI HWC float tensor) ----------
        img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # --- Estimate depth using a lightweight PyTorch model -------------
        depth_np = self._estimate_depth_cuda(img_np, depth_estimator)

        # Normalise
        img_f = img_np.astype(np.float32) / 255.0
        dep_f = depth_np.astype(np.float32)
        if dep_f.max() > 1.5:
            dep_f = dep_f / 255.0

        renderer = cuda_renderer.CudaDepthFlowRenderer(img_f, dep_f)
        pbar = ProgressBar(int(duration * fps))

        def _progress(cur, total):
            pbar.update_absolute(cur, total)

        # Capture frames directly during render to avoid re-decoding mp4
        max_cap = max_frames_export if max_frames_export > 0 else -1

        # Check if CUDA renderer supports inpaint parameters
        import inspect
        render_sig = inspect.signature(cuda_renderer.CudaDepthFlowRenderer.render_video)
        has_inpaint = 'enable_inpaint' in render_sig.parameters

        inpaint_kwargs = {}
        if has_inpaint:
            inpaint_kwargs = dict(
                enable_inpaint=cuda_enable_inpaint,
                inpaint_threshold=cuda_inpaint_threshold,
                inpaint_blur=cuda_inpaint_blur,
                enable_aa=cuda_enable_aa,
            )
        else:
            print("[DepthFlow] ⚠ CUDA renderer found but does not support "
                  "inpaint parameters. Please update GentlemanHu/DepthFlow fork.")

        renderer.render_video(
            output_path=output_path,
            render_w=target_width,
            render_h=target_height,
            fps=fps,
            duration=duration,
            ssaa=ssaa,
            quality_pct=quality,
            camera_movement=camera_movement,
            intensity=movement_intensity,
            smooth=movement_smooth,
            loop=movement_loop,
            reverse=movement_reverse,
            phase=movement_phase,
            steady_depth=steady_depth,
            isometric_val=isometric,
            codec=video_codec,
            output_format=output_format,
            progress_cb=_progress,
            capture_frames=max_cap if output_frames else 0,
            **inpaint_kwargs,
        )

        # Stash captured frames for _finalize_output to use
        self._cuda_captured_frames = getattr(renderer, 'captured_frames', None)

        if not os.path.exists(output_path):
            return False

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[DepthFlow] ✓ CUDA render OK — {size_mb:.1f} MB")
        return True

    @staticmethod
    def _estimate_depth_cuda(img_np: np.ndarray, estimator: str = "da2") -> np.ndarray:
        """Estimate a depth map using DepthAnythingV2 in the ComfyUI process.

        Falls back to a simple Laplacian mock if the model is unavailable, so
        the rendering pipeline can still be tested.
        """
        try:
            # Try using transformers pipeline (widely available)
            from transformers import pipeline as hf_pipeline
            model_ids = {
                "da2": "depth-anything/Depth-Anything-V2-Small-hf",
                "da1": "LiheYoung/depth-anything-small-hf",
            }
            model_id = model_ids.get(estimator, model_ids["da2"])
            print(f"[DepthFlow] Estimating depth with {model_id} ...")
            pipe = hf_pipeline("depth-estimation", model=model_id, device=0)
            from PIL import Image as _PILImage
            pil_img = _PILImage.fromarray(img_np)
            result = pipe(pil_img)
            depth_pil = result["depth"]
            depth = np.array(depth_pil, dtype=np.float32)
            if depth.max() > 0:
                depth = depth / depth.max()
            print(f"[DepthFlow] ✓ Depth estimated: {depth.shape}")
            return depth
        except Exception as e:
            print(f"[DepthFlow] ⚠ HF depth estimation failed ({e}), trying torch.hub ...")

        try:
            # Try torch hub
            model = torch.hub.load("hustvl/Depth-Anything-V2", "depth_anything_v2_vits",
                                    pretrained=True, trust_repo=True)
            model = model.cuda().eval()
            from PIL import Image as _PILImage
            import torchvision.transforms as T
            pil_img = _PILImage.fromarray(img_np)
            t = T.Compose([T.Resize(518), T.CenterCrop(518),
                           T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            inp = t(pil_img).unsqueeze(0).cuda()
            with torch.inference_mode():
                depth = model(inp).squeeze().cpu().numpy()
            if depth.max() > 0:
                depth = depth / depth.max()
            return depth
        except Exception as e2:
            print(f"[DepthFlow] ⚠ torch.hub depth failed ({e2}), using luminance fallback")

        # Fallback: luminance-based pseudo-depth (still renders, less 3D)
        gray = np.mean(img_np.astype(np.float32), axis=2)
        depth = 1.0 - (gray / max(gray.max(), 1.0))
        return depth


class VideoPreview:
    """Universal video preview node supporting local and remote URLs"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "auto_download": ("BOOLEAN", {"default": True}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview_video"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    
    def is_url(self, path):
        """Check if path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def preview_video(self, video_path, auto_download=True, force_refresh=False):
        """Preview video from local path or URL"""
        
        if not video_path or not video_path.strip():
            return {"ui": {"text": ["No video path provided"]}, "result": ("",)}
        
        video_path = video_path.strip()
        
        # Handle URL
        if self.is_url(video_path):
            print(f"[VideoPreview] Remote URL detected: {video_path}")
            
            cache_path = video_manager.get_cache_path(video_path)
            
            # Force refresh - delete cache
            if force_refresh and os.path.exists(cache_path):
                print(f"[VideoPreview] Force refresh - removing cache")
                os.remove(cache_path)
            
            # Check cache
            if video_manager.is_cached(video_path):
                print(f"[VideoPreview] Using cached video")
                local_path = cache_path
                status = "cached"
            elif auto_download:
                print(f"[VideoPreview] Starting async download")
                local_path = video_manager.download_async(video_path)
                download_status = video_manager.get_status(video_path)
                status = download_status["status"]
            else:
                print(f"[VideoPreview] Auto-download disabled")
                return {
                    "ui": {"text": ["Video not cached. Enable auto_download to download."]},
                    "result": ("",)
                }
            
            # Get filename from cache path
            filename = os.path.basename(local_path)
            
            return {
                "ui": {
                    "video": [{
                        "filename": filename,
                        "subfolder": "video_cache",
                        "type": "temp"
                    }],
                    "text": [f"Remote video (cached): {video_path}"]
                },
                "result": (local_path,)
            }
        
        # Handle local path
        else:
            # Convert to absolute path
            if not os.path.isabs(video_path):
                # Try relative to output directory
                abs_path = os.path.join(folder_paths.get_output_directory(), video_path)
                if not os.path.exists(abs_path):
                    # Try relative to input directory
                    abs_path = os.path.join(folder_paths.get_input_directory(), video_path)
                if not os.path.exists(abs_path):
                    # Try as-is
                    abs_path = os.path.abspath(video_path)
            else:
                abs_path = video_path
            
            # Check if file exists
            if not os.path.exists(abs_path):
                return {
                    "ui": {"text": [f"Video file not found: {abs_path}"]},
                    "result": ("",)
                }
            
            file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            print(f"[VideoPreview] Local video: {abs_path} ({file_size_mb:.2f} MB)")
            
            # Determine subfolder and filename
            output_dir = folder_paths.get_output_directory()
            if abs_path.startswith(output_dir):
                # File is in output directory
                rel_path = os.path.relpath(abs_path, output_dir)
                subfolder = os.path.dirname(rel_path)
                filename = os.path.basename(rel_path)
                file_type = "output"
            else:
                # File is elsewhere, copy to temp
                temp_dir = folder_paths.get_temp_directory()
                filename = os.path.basename(abs_path)
                temp_path = os.path.join(temp_dir, filename)
                
                # Copy if not already in temp
                if abs_path != temp_path:
                    import shutil
                    shutil.copy2(abs_path, temp_path)
                    abs_path = temp_path
                
                subfolder = ""
                file_type = "temp"
            
            return {
                "ui": {
                    "video": [{
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": file_type
                    }],
                    "text": [f"Local video: {abs_path} ({file_size_mb:.2f} MB)"]
                },
                "result": (abs_path,)
            }


class VideoPathInput:
    """Simple video path input node that passes through to VideoPreview"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "pass_through"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    
    def pass_through(self, video):
        """Pass through video path"""
        return (video,)


class VideoCacheManager:
    """Manage video cache - clear, list, etc."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["list", "clear_all", "clear_old"],{"default": "list"}),
                "days_old": ("INT", {"default": 7, "min": 1, "max": 365}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    FUNCTION = "manage_cache"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    
    def manage_cache(self, action="list", days_old=7):
        """Manage video cache"""
        
        if not os.path.exists(VIDEO_CACHE_DIR):
            return {"ui": {"message": "Cache directory does not exist"}, "result": ("",)}
        
        cache_files = list(Path(VIDEO_CACHE_DIR).glob("*"))
        
        if action == "list":
            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            info = f"Cache directory: {VIDEO_CACHE_DIR}\n"
            info += f"Total files: {len(cache_files)}\n"
            info += f"Total size: {total_size_mb:.2f} MB\n\n"
            
            for f in cache_files[:10]:  # Show first 10
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    info += f"- {f.name} ({size_mb:.2f} MB)\n"
            
            if len(cache_files) > 10:
                info += f"\n... and {len(cache_files) - 10} more files"
            
            print(f"[VideoCacheManager] {info}")
            return {"ui": {"message": info}, "result": (info,)}
        
        elif action == "clear_all":
            count = 0
            for f in cache_files:
                try:
                    if f.is_file():
                        f.unlink()
                        count += 1
                except Exception as e:
                    print(f"[VideoCacheManager] Failed to delete {f}: {e}")
            
            info = f"Cleared {count} cached videos"
            print(f"[VideoCacheManager] {info}")
            return {"ui": {"message": info}, "result": (info,)}
        
        elif action == "clear_old":
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            count = 0
            
            for f in cache_files:
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff_time:
                        f.unlink()
                        count += 1
                except Exception as e:
                    print(f"[VideoCacheManager] Failed to delete {f}: {e}")
            
            info = f"Cleared {count} videos older than {days_old} days"
            print(f"[VideoCacheManager] {info}")
            return {"ui": {"message": info}, "result": (info,)}
        
        return {"ui": {"message": "Unknown action"}, "result": ("",)}


# Node registration
NODE_CLASS_MAPPINGS = {
    "DepthFlowGenerator": DepthFlowGenerator,
    "VideoPreview": VideoPreview,
    "VideoPathInput": VideoPathInput,
    "VideoCacheManager": VideoCacheManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthFlowGenerator": "DepthFlow Generator 🌊",
    "VideoPreview": "Video Preview 🎬",
    "VideoPathInput": "Video Path Input 📁",
    "VideoCacheManager": "Video Cache Manager 🗑️",
}
