"""
Qwen Save Utilities - High Performance Edition
COMPLETE SUITE: Date, Text, Image, Concatenation
Optimized for: Local File Management, Metadata, Batch Processing, and Validation Safety.

Optimized by: Principal Python Performance Engineer
"""

import os
import re
import json
import torch
import numpy as np
import logging
from PIL import Image, PngImagePlugin
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Union

# --- Logging Setup ---
logger = logging.getLogger("QwenSave")

# --- Helper Functions ---

def get_next_index(directory: Path, filename_prefix: str) -> int:
    """
    Scans the directory for the highest existing index for a given prefix.
    Returns the next available integer (1-based).
    """
    max_idx = 0
    if not directory.exists():
        return 1
    
    prefix_with_sep = f"{filename_prefix}_"
    prefix_len = len(prefix_with_sep)

    try:
        # scandir is faster than listdir for large directories
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.startswith(prefix_with_sep):
                stem = Path(entry.name).stem
                suffix = stem[prefix_len:]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
    except OSError as e:
        logger.warning(f"Directory scan failed: {e}")
        return 1
        
    return max_idx + 1

def sanitize_filename(filename: str) -> str:
    """Removes extensions and invalid characters from filenames."""
    return Path(filename).stem

# --- Node Classes ---

class Qwen_DateGenerator:
    """
    Generates formatted date strings for path/filename organization.
    """
    FORMAT_MAP = {
        "yyyy-mm-dd-hh-mm-ss": "%Y-%m-%d-%H-%M-%S",
        "yyyy-mm-dd": "%Y-%m-%d",
        "yy-mm-dd": "%y-%m-%d",
        "yyyy_mm_dd": "%Y_%m_%d",
        "hh-mm-ss": "%H-%M-%S",
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset_format": (list(cls.FORMAT_MAP.keys()), {"default": "yyyy-mm-dd"}),
                "use_custom": ("BOOLEAN", {"default": False}),
                "custom_format": ("STRING", {"default": "%Y/%m/%d"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("date_string",)
    FUNCTION = "get_date"
    CATEGORY = "Qwen-IO"

    def get_date(self, preset_format: str, use_custom: bool, custom_format: str) -> Tuple[str]:
        fmt = custom_format if use_custom else self.FORMAT_MAP.get(preset_format, "%Y-%m-%d")
        try:
            date_str = datetime.now().strftime(fmt)
        except ValueError:
            logger.error(f"Invalid date format '{fmt}'. Falling back to default.")
            date_str = datetime.now().strftime("%Y-%m-%d")
        return (date_str,)



class Qwen_TextConcatenate:
    """
    Advanced Text Concatenator - High Performance Edition
    Features:
    - Dynamic Integer Input (1-12 slots).
    - Auto-generated slots via JS.
    - Robust logic that safely ignores disconnected or hidden slots.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {
                "input_count": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1, "display": "number"}),
                "seperator": ("STRING", {"default": " ", "multiline": False}),
                "clean_whitespace": ("BOOLEAN", {"default": True, "tooltip": "Fixes spaces, removes ' . ' and deduplicates ',,' or '..'"}),
                "search": ("STRING", {"default": "", "multiline": False}),
                "replace": ("STRING", {"default": "", "multiline": False}),
                "use_regex": ("BOOLEAN", {"default": False}),
            },
            "optional": {}
        }
        
        # Pre-define all possible slots (1-12) as optional.
        # This allows the backend to accept them if the frontend provides them,
        # but prevents errors if they are missing.
        for i in range(1, 5):
            inputs["optional"][f"text_{i}"] = ("STRING", {"forceInput": True})
            
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_text",)
    FUNCTION = "process_text"
    CATEGORY = "Qwen-IO"

    def process_text(self, input_count: int, seperator: str, clean_whitespace: bool, 
                     search: str, replace: str, use_regex: bool, **kwargs) -> Tuple[str]:
        
        # 1. Clamp Active Range
        # We trust the widget, but clamp for safety.
        active_count = max(1, min(int(input_count), 12))
        
        # 2. Collect Inputs
        # Efficiently gather only the inputs that fall within the active range.
        # This ignores any 'ghost' inputs that might still be in kwargs but hidden in UI.
        collected_texts = []
        for i in range(1, active_count + 1):
            # Using get() handles cases where the slot is completely removed/missing
            val = kwargs.get(f"text_{i}")
            
            if val is not None:
                val = str(val)
                if clean_whitespace:
                    val = val.strip()
                if val:
                    collected_texts.append(val)

        # 3. Handle Delimiter Escapes
        real_delimiter = seperator.replace("\\n", "\n").replace("\\t", "\t")

        # 4. Join
        final_text = real_delimiter.join(collected_texts)

        # 5. Grammar & Regex Cleaning
        if clean_whitespace:
            # Collapse multiple spaces
            final_text = re.sub(r'\s+', ' ', final_text).strip()
            # Fix punctuation spacing " . " -> ". "
            final_text = re.sub(r'\s+([,.?!:;])', r'\1', final_text)
            # Deduplicate punctuation ",," -> ","
            final_text = re.sub(r'([,.?!:;])\1+', r'\1', final_text)

        if search:
            try:
                if use_regex:
                    final_text = re.sub(search, replace, final_text)
                else:
                    final_text = final_text.replace(search, replace)
            except re.error as e:
                logger.error(f"Regex Error in Qwen_TextConcatenate: {e}")
                
        return (final_text,)


class Qwen_TextSave:
    """
    Saves text to file with auto-indexing and existence checks.
    """
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {
                # The name "input_count" MUST match the JS search
                "input_count": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1, "display": "number"}), 
                "seperator": ("STRING", {"default": " ", "multiline": False}),
                "clean_whitespace": ("BOOLEAN", {"default": True}),
                "search": ("STRING", {"default": "", "multiline": False}),
                "replace": ("STRING", {"default": "", "multiline": False}),
                "use_regex": ("BOOLEAN", {"default": False}),
            },
            "optional": {}
        }
        
        # Pre-define all slots so Python doesn't reject them before JS removes them
        for i in range(1, 13):
            inputs["optional"][f"text_{i}"] = ("STRING", {"forceInput": True})
            
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_text"
    CATEGORY = "Qwen-IO"
    OUTPUT_NODE = True

    def save_text(self, text: str, path: str, filename: str, overwrite: bool, 
                  auto_index: bool, manual_index: int = -1, subfolder: str = "") -> Dict[str, Any]:
        
        clean_filename = sanitize_filename(filename)
        base_dir = Path(path)
        if subfolder.strip():
            base_dir = base_dir / subfolder.strip()
        
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return {"ui": {"text": [f"Error creating directory: {e}"]}, "result": ("",)}

        index_suffix = ""
        if auto_index:
            idx = get_next_index(base_dir, clean_filename)
            index_suffix = f"_{idx:05d}"
        elif manual_index >= 0:
            index_suffix = f"_{manual_index:05d}"
        
        full_path = base_dir / f"{clean_filename}{index_suffix}.txt"

        if not overwrite and full_path.exists() and not auto_index:
             return {"ui": {"text": [f"⚠️ Skipped (Exists): {full_path}"]}, "result": (str(full_path),)}

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(text)
        except IOError as e:
            return {"ui": {"text": [f"Error writing file: {e}"]}, "result": ("",)}

        return {"ui": {"text": [f"Saved: {full_path}"]}, "result": (str(full_path),)}

# --- Main Node Class ---

class Qwen_ImageSave:
    """
    Saves image batches with hybrid metadata support (WAS-Suite & A1111 compatibility).
    
    Features: 
    - High-Performance NumPy conversion.
    - Dual Metadata Modes:
      1. 'ComfyUI' (WAS-Style): Best for loading workflows back into ComfyUI.
      2. 'A1111/Civitai' (Qwen-Style): Adds 'parameters' text/tag for external parsers.
    - WebP Lossless & Optimization support.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "./output"}),
                "filename": ("STRING", {"default": "output_image"}),
                "extension": (["png", "jpg", "webp", "bmp", "tiff"], {"default": "webp"}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Quality 1-100. Affects JPG and WebP."}),
                "lossless": ("BOOLEAN", {"default": False, "tooltip": "Only affects WebP."}),
                "optimize_image": ("BOOLEAN", {"default": True, "tooltip": "Perform optimization pass (smaller size, slower save)."}),
                "auto_index": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Save Prompt and Workflow metadata."}),
                "meta_format": (["ComfyUI", "A1111/Civitai"], {"default": "ComfyUI"}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "manual_index": ("INT", {"default": -1, "min": -1}),
                "subfolder": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "save_images"
    CATEGORY = "Qwen-IO"
    OUTPUT_NODE = True

    def save_images(self, image: torch.Tensor, path: str, filename: str, extension: str, 
                    quality: int, lossless: bool, optimize_image: bool, auto_index: bool, 
                    save_metadata: bool, meta_format: str, show_preview: bool, overwrite: bool, 
                    manual_index: int = -1, subfolder: str = "",
                    prompt=None, extra_pnginfo=None) -> Dict[str, Any]:
        
        clean_filename = sanitize_filename(filename)
        
        # Path Resolution
        if os.path.isabs(path):
            base_dir = Path(path)
        else:
            base_dir = Path(os.getcwd()) / path

        if subfolder.strip():
            base_dir = base_dir / subfolder.strip()
        
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             return {"ui": {"text": [f"Error creating dir: {e}"]}, "result": ("",)}

        # Indexing Setup
        current_idx = 1
        if auto_index:
            current_idx = get_next_index(base_dir, clean_filename)
        elif manual_index >= 0:
            current_idx = manual_index

        saved_paths = []
        preview_images = []

        # Loop through batch
        for i, img_tensor in enumerate(image):
            suffix = ""
            if auto_index or manual_index >= 0 or len(image) > 1:
                suffix = f"_{current_idx:05d}"
            
            full_path = base_dir / f"{clean_filename}{suffix}.{extension}"
            
            # Increment index
            if auto_index or manual_index >= 0 or len(image) > 1:
                current_idx += 1

            # Overwrite Check
            if not overwrite and full_path.exists():
                if not auto_index and len(image) == 1:
                    logger.warning(f"File exists and overwrite=False: {full_path}")
                    saved_paths.append(str(full_path))
                    continue

            # Tensor -> Numpy -> PIL (High Performance)
            img_np = (255. * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # --- Metadata Logic ---
            exif_data = None
            png_metadata = PngImagePlugin.PngInfo()

            if save_metadata:
                # 1. Setup PNG Metadata (Universal Base)
                if prompt:
                    png_metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for key, value in extra_pnginfo.items():
                        png_metadata.add_text(key, json.dumps(value))
                
                # A1111 Compatibility for PNG
                if meta_format == "A1111/Civitai" and prompt:
                    png_metadata.add_text("parameters", f"Prompt: {json.dumps(prompt)}")

                # 2. Setup WebP/JPG Metadata (EXIF)
                if extension in ['webp', 'jpg', 'jpeg']:
                    img_exif = pil_img.getexif()

                    # MODE A: WAS-Suite Style (Default/ComfyUI)
                    # Best for reloading workflows in ComfyUI
                    if meta_format == "ComfyUI":
                        if prompt:
                            # 0x010f = ImageDescription / Make
                            img_exif[0x010f] = "Prompt:" + json.dumps(prompt)
                        if extra_pnginfo:
                            workflow_metadata = ""
                            for x in extra_pnginfo:
                                workflow_metadata += json.dumps(extra_pnginfo[x])
                            # 0x010e = ImageDescription / Model
                            img_exif[0x010e] = "Workflow:" + workflow_metadata
                    
                    # MODE B: Qwen/A1111 Style
                    # Best for Civitai/A1111 detection (Single JSON container)
                    elif meta_format == "A1111/Civitai":
                        meta_dict = {}
                        if prompt:
                            meta_dict["prompt"] = prompt
                        if extra_pnginfo:
                            for key, value in extra_pnginfo.items():
                                meta_dict[key] = value
                        
                        # Add the specific "parameters" key A1111 parsers look for
                        if prompt:
                            meta_dict["parameters"] = f"Prompt: {json.dumps(prompt)}"
                        
                        if meta_dict:
                            # Dump entire dict to 0x010e
                            img_exif[0x010e] = json.dumps(meta_dict)

                    exif_data = img_exif.tobytes()

            # --- Save Execution ---
            try:
                save_args = {
                    "optimize": optimize_image,
                }

                if extension == 'webp':
                    save_args["quality"] = quality
                    save_args["lossless"] = lossless
                    if exif_data:
                        save_args["exif"] = exif_data
                        
                elif extension == 'png':
                    save_args["compress_level"] = 4 
                    if png_metadata:
                        save_args["pnginfo"] = png_metadata
                        
                elif extension in ['jpg', 'jpeg']:
                    save_args["quality"] = quality
                    if pil_img.mode == 'RGBA':
                        pil_img = pil_img.convert('RGB')
                    # JPG EXIF support is limited in Pillow without specialized handling,
                    # but we pass the bytes if generated.
                    if exif_data:
                        save_args["exif"] = exif_data
                
                elif extension == 'tiff':
                    save_args["quality"] = quality

                pil_img.save(str(full_path), **save_args)
                saved_paths.append(str(full_path))
                
                if show_preview:
                    preview_images.append({
                        "filename": full_path.name,
                        "subfolder": str(subfolder) if subfolder else "",
                        "type": "output"
                    })

            except Exception as e:
                logger.error(f"Failed to save {full_path}: {e}")

        # Return results
        result_dict = {"ui": {}, "result": (saved_paths,)}
        if show_preview and preview_images:
            result_dict["ui"]["images"] = preview_images
        
        if saved_paths:
            display_text = f"Saved {len(saved_paths)} images to {base_dir}"
            result_dict["ui"]["text"] = [display_text]
        else:
             result_dict["ui"]["text"] = ["No images saved."]

        return result_dict

# --- Registration ---

NODE_CLASS_MAPPINGS = {
    "Qwen_DateGenerator": Qwen_DateGenerator,
    "Qwen_TextConcatenate": Qwen_TextConcatenate,
    "Qwen_TextSave": Qwen_TextSave,
    "Qwen_ImageSave": Qwen_ImageSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen_DateGenerator": "RM-Date Generator",
    "Qwen_TextConcatenate": "RM-Text Concatenate",
    "Qwen_TextSave": "RM-Text Save",
    "Qwen_ImageSave": "RM-Image Save",
}