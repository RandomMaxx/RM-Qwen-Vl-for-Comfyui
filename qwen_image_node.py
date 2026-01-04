import os
import json
import math
import asyncio
import torch
import numpy as np
from typing import Any, List, Dict, Tuple
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps
from aiohttp import web
from functools import lru_cache

import folder_paths
import server

# --- CONFIGURATION ---
NODE_FILE_PATH = Path(__file__).parent.resolve()
HISTORY_FILE = NODE_FILE_PATH / "qwen_history.json"
MAX_HISTORY_ITEMS = 15
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

RES_PRESETS = tuple(str(i) for i in range(128, 2049, 64))
MOD_PRESETS = ("1", "2", "4", "8", "16", "32", "64")
RESAMPLING_METHODS = ["bicubic", "bilinear", "nearest", "box", "hamming", "lanczos"]

# --- HISTORY MANAGER ---
class HistoryManager:
    @staticmethod
    def load() -> List[str]:
        if not HISTORY_FILE.exists(): return []
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            paths = data.get("paths", [])
            valid_paths = [p for p in paths if Path(p).exists()]
            if len(valid_paths) != len(paths): HistoryManager._write(valid_paths)
            return valid_paths
        except: return []

    @staticmethod
    def save(path_str: str) -> List[str]:
        if not path_str: return HistoryManager.load()
        try:
            abs_path = Path(path_str).resolve()
            if not abs_path.exists(): return HistoryManager.load()
            paths = HistoryManager.load()
            s_path = str(abs_path)
            if s_path in paths: paths.remove(s_path)
            paths.insert(0, s_path)
            paths = paths[:MAX_HISTORY_ITEMS]
            HistoryManager._write(paths)
            return paths
        except: return HistoryManager.load()

    @staticmethod
    def _write(paths: List[str]):
        HISTORY_FILE.write_text(json.dumps({"paths": paths}, indent=2), encoding="utf-8")

# --- FILE SCANNER ---
def scan_directory_filtered(folder_root: str, recursive: bool, filter_str: str = "") -> List[str]:
    root_path = Path(folder_root)
    if not root_path.exists() or not root_path.is_dir(): return []
    
    files_found = []
    f_lower = filter_str.lower()
    
    try:
        if recursive:
            for root, _, files in os.walk(folder_root):
                for name in files:
                    if not name.startswith(".") and Path(name).suffix.lower() in VALID_EXTENSIONS:
                        rel = os.path.relpath(os.path.join(root, name), folder_root).replace("\\", "/")
                        if not filter_str or f_lower in rel.lower():
                            files_found.append(rel)
        else:
            with os.scandir(folder_root) as entries:
                for entry in entries:
                    if entry.is_file() and not entry.name.startswith(".") and Path(entry.name).suffix.lower() in VALID_EXTENSIONS:
                        if not filter_str or f_lower in entry.name.lower():
                            files_found.append(entry.name)
    except Exception: pass
    files_found.sort()
    return files_found

# --- API ROUTES ---
async def api_get_files(request: web.Request) -> web.Response:
    query = request.rel_url.query
    path_str, filter_str = query.get("path", "input"), query.get("filter", "")
    recursive = query.get("recursive", "false") == "true"
    
    # Resolve Path
    base_input = Path(folder_paths.get_input_directory())
    if Path(path_str).is_absolute():
        root_path = Path(path_str)
    else:
        root_path = (base_input / path_str.replace("input/", "").replace("input", "")).resolve()
    
    # Side Effect: Update History on successful scan request
    if root_path.exists():
        HistoryManager.save(str(root_path))
        
    files = await asyncio.to_thread(scan_directory_filtered, str(root_path), recursive, filter_str)
    return web.json_response({"files": files})

async def api_live_preview(request: web.Request) -> web.Response:
    query = request.rel_url.query
    path_str, filename = query.get("path", "input"), query.get("filename", "")
    if not filename or filename.startswith("("): return web.Response(status=400)
    
    base_input = Path(folder_paths.get_input_directory())
    if Path(path_str).is_absolute():
        root_path = Path(path_str)
    else:
        root_path = (base_input / path_str.replace("input/", "").replace("input", "")).resolve()
        
    file_path = root_path / filename
    if not file_path.exists(): return web.Response(status=404)
    
    def _get_preview():
        try:
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((512, 512), Image.Resampling.BILINEAR)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=80)
                return buf.getvalue()
        except: return None

    img_bytes = await asyncio.to_thread(_get_preview)
    return web.Response(body=img_bytes, content_type="image/jpeg") if img_bytes else web.Response(status=500)

async def api_add_history(request: web.Request) -> web.Response:
    data = await request.json()
    path = data.get("path", "")
    updated_list = HistoryManager.save(path)
    return web.json_response({"history": updated_list})

if hasattr(server, "PromptServer") and server.PromptServer.instance is not None:
    s = server.PromptServer.instance
    s.app.router.add_get("/qwen/files", api_get_files)
    s.app.router.add_get("/qwen/live_preview", api_live_preview)
    s.app.router.add_post("/qwen/history/add", api_add_history)

# --- THE NODE ---
class Qwen3VL_ImageLoader:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Build Quick Select List
        history = HistoryManager.load()
        input_dir = Path(folder_paths.get_input_directory())
        subfolders = []
        if input_dir.exists():
            try:
                # Add root if valid
                subfolders.append("input")
                # Add subfolders
                for root, dirs, _ in os.walk(input_dir):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for d in dirs:
                        rel = os.path.relpath(os.path.join(root, d), input_dir).replace("\\", "/")
                        subfolders.append(f"input/{rel}")
            except: pass
        subfolders.sort()
        
        quick_select_items = ["(Select to Auto-Fill)"] + history + ["--- Local Input Folders ---"] + subfolders

        return {
            "required": {
                "directory": ("STRING", {"default": "input"}),
                "quick_select": (quick_select_items, ),
                "filter": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                "filename": (["(Wait for scan...)"],), 
                "recursive": ("BOOLEAN", {"default": False}),
                "resize_image": ("BOOLEAN", {"default": True}),
                "resolution": (RES_PRESETS, {"default": "512"}),
                "modulo": (MOD_PRESETS, {"default": "4"}),
                "resampling": (RESAMPLING_METHODS,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("resized_image", "original_image", "width", "height", "filename", "full_path", "current_index")
    FUNCTION = "load_image"
    CATEGORY = "Qwen3-VL/Image"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs): return True

    def load_image(self, directory: str, quick_select: str, filter: str, index: int, filename: str,
                   recursive: bool, resize_image: bool, resolution: str, modulo: str, resampling: str) -> Tuple[Any, ...]:
        
        base_input = Path(folder_paths.get_input_directory())
        if Path(directory).is_absolute():
            root_path = Path(directory)
        else:
            root_path = (base_input / directory.replace("input/", "").replace("input", "")).resolve()

        # Update History
        HistoryManager.save(str(root_path))

        files = scan_directory_filtered(str(root_path), recursive, filter)
        if not files: 
            # Fallback if no files match filter but directory exists
            if root_path.exists():
                 raise ValueError(f"No files match filter '{filter}' in {root_path}")
            raise ValueError(f"Path not found: {root_path}")

        max_files = len(files)
        target_idx = (index - 1) % max_files
        selected_file = files[target_idx]
        full_path = root_path / selected_file
        
        # Calculate current index (1-based)
        current_index = target_idx + 1

        with Image.open(full_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            orig_w, orig_h = img.size
            orig_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None,]
            
            if resize_image:
                res, mod = int(resolution), int(modulo)
                scale = math.sqrt((res**2) / (orig_w * orig_h))
                nw = max(mod, int(round(orig_w * scale / mod) * mod))
                nh = max(mod, int(round(orig_h * scale / mod) * mod))
                
                resample_map = {
                    "bicubic": Image.Resampling.BICUBIC,
                    "bilinear": Image.Resampling.BILINEAR,
                    "nearest": Image.Resampling.NEAREST,
                    "box": Image.Resampling.BOX,
                    "hamming": Image.Resampling.HAMMING,
                    "lanczos": Image.Resampling.LANCZOS
                }
                m = resample_map.get(resampling, Image.Resampling.BICUBIC)
                
                resized_img = img.resize((nw, nh), m)
                resized_t = torch.from_numpy(np.array(resized_img, dtype=np.float32) / 255.0)[None,]
                
                return (resized_t, orig_t, nw, nh, selected_file, str(full_path), current_index)
            
            return (orig_t, orig_t, orig_w, orig_h, selected_file, str(full_path), current_index)

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

NODE_CLASS_MAPPINGS = { "Qwen3VL_ImageLoader": Qwen3VL_ImageLoader }
NODE_DISPLAY_NAME_MAPPINGS = { "Qwen3VL_ImageLoader": "RM-Qwen3-VL Image Loader" }