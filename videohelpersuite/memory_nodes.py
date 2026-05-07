import gc

import torch

from .logger import logger


class MemoryCleanupPassthrough:
    class Any(str):
        def __ne__(self, other):
            return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "unload_models": ("BOOLEAN", {"default": True}),
                "free_cuda_cache": ("BOOLEAN", {"default": True}),
                "gc_collect": ("BOOLEAN", {"default": True}),
                "log_memory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (Any("*"), "STRING")
    RETURN_NAMES = ("value", "info")
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/utils"
    FUNCTION = "cleanup"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _memory_snapshot(self) -> str:
        parts = []
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                parts.append(
                    "cuda="
                    f"free:{free / 1024 ** 3:.2f}GiB/"
                    f"total:{total / 1024 ** 3:.2f}GiB "
                    f"allocated:{allocated / 1024 ** 3:.2f}GiB "
                    f"reserved:{reserved / 1024 ** 3:.2f}GiB"
                )
            except Exception as exc:
                parts.append(f"cuda=unavailable_snapshot:{exc}")
        meminfo = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, raw = line.split(":", 1)
                    meminfo[key] = int(raw.strip().split()[0]) * 1024
            available = meminfo.get("MemAvailable")
            total = meminfo.get("MemTotal")
            if available and total:
                parts.append(f"ram=available:{available / 1024 ** 3:.2f}GiB/total:{total / 1024 ** 3:.2f}GiB")
        except Exception:
            pass
        return " | ".join(parts) if parts else "memory snapshot unavailable"

    def cleanup(self, value, unload_models=True, free_cuda_cache=True, gc_collect=True, log_memory=True):
        before = self._memory_snapshot()
        if unload_models:
            try:
                import comfy.model_management

                comfy.model_management.unload_all_models()
            except Exception as exc:
                logger.warn(f"[VHS Memory Cleanup] unload_all_models failed: {exc}")
        if gc_collect:
            gc.collect()
        if free_cuda_cache and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as exc:
                logger.warn(f"[VHS Memory Cleanup] cuda cache cleanup failed: {exc}")
        if unload_models or free_cuda_cache:
            try:
                import comfy.model_management

                comfy.model_management.soft_empty_cache(force=True)
            except Exception:
                pass
        after = self._memory_snapshot()
        info = f"before: {before}\nafter: {after}"
        if log_memory:
            logger.info(f"[VHS Memory Cleanup] {info.replace(chr(10), ' | ')}")
        return (value, info)
