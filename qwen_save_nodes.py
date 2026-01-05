"""
Qwen Save Utilities - High Performance Edition
Simple, robust nodes for Text/Image Saving and Viewing.
Optimized for local file management without external bloat.

Optimized by: Principal Python Performance Engineer
"""

import os
import json
import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# --- Logging Setup ---
logger = logging.getLogger("QwenSave")

class Qwen_TextView:
    """
    Simple node to preview text directly on the canvas.
    Acts as a pass-through, so it can be placed anywhere in the chain.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "view_text"
    CATEGORY = "Qwen-IO"
    OUTPUT_NODE = True

    def view_text(self, text):
        # We limit the preview to 5000 characters to prevent frontend lag
        # if a massive text file is passed.
        preview = text
        if len(preview) > 5000:
            preview = preview[:5000] + "\n... [Text Truncated for Performance]"

        return {"ui": {"text": [preview]}, "result": (text,)}


class Qwen_TextSave:
    """
    Saves a text string to a local file.
    Features: 
    - Index appending (00001)
    - Automatic subfolder creation
    - On-Node Text Preview
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "path": ("STRING", {"default": "./output"}),
                "filename": ("STRING", {"default": "output_text"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "index": ("INT", {"default": -1, "min": -1, "max": 999999, "tooltip": "-1 to disable index appending"}),
                "subfolder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_text"
    CATEGORY = "Qwen-IO"
    OUTPUT_NODE = True

    def save_text(self, text, path, filename, overwrite, index=-1, subfolder=""):
        # 1. Path Construction
        base_dir = Path(path)
        if subfolder.strip():
            base_dir = base_dir / subfolder.strip()
        
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create directory '{base_dir}': {e}")

        # 2. Filename Construction
        name_part = filename.strip()
        if index >= 0:
            name_part = f"{name_part}_{index:05d}"
        
        full_path = base_dir / f"{name_part}.txt"

        # 3. Overwrite Protection
        if not overwrite and full_path.exists():
            if index == -1:
                counter = 1
                while full_path.exists():
                    full_path = base_dir / f"{name_part}_{counter:05d}.txt"
                    counter += 1
            else:
                logger.warning(f"File '{full_path}' exists and overwrite is False. Skipping save.")
                return {
                    "ui": {"text": [f"⚠️ Skipped (Exists):\n{full_path}"]}, 
                    "result": (str(full_path),)
                }

        # 4. Write to Disk
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Saved text to: {full_path}")
        except Exception as e:
            raise IOError(f"Failed to save text file: {e}")

        # 5. Return with UI Preview (Limit preview to 1000 chars)
        preview_text = f"Saved to: {full_path}\n\nContent Preview:\n{text[:1000]}"
        if len(text) > 1000:
            preview_text += "..."
            
        return {"ui": {"text": [preview_text]}, "result": (str(full_path),)}


class Qwen_ImageSave:
    """
    Saves an image tensor to disk.
    Features:
    - PNG/JPG/WEBP support
    - Lossless/Compression toggles
    - On-Node Path Preview
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "./output"}),
                "filename": ("STRING", {"default": "output_image"}),
                "extension": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "100 = Best Quality / Lowest Compression"}),
                "lossless": ("BOOLEAN", {"default": False, "tooltip": "Overrides quality for supported formats (WEBP/PNG)"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "index": ("INT", {"default": -1, "min": -1, "max": 999999, "tooltip": "-1 to disable index appending"}),
                "subfolder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_image"
    CATEGORY = "Qwen-IO"
    OUTPUT_NODE = True

    def save_image(self, image, path, filename, extension, quality, lossless, overwrite, index=-1, subfolder=""):
        # 1. Path Construction
        base_dir = Path(path)
        if subfolder.strip():
            base_dir = base_dir / subfolder.strip()
        
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create directory '{base_dir}': {e}")

        # 2. Filename Construction
        name_part = filename.strip()
        if index >= 0:
            name_part = f"{name_part}_{index:05d}"
        
        full_path = base_dir / f"{name_part}.{extension}"

        # 3. Overwrite Logic
        if not overwrite and full_path.exists():
            if index == -1:
                counter = 1
                while full_path.exists():
                    full_path = base_dir / f"{name_part}_{counter:05d}.{extension}"
                    counter += 1
            else:
                logger.warning(f"File '{full_path}' exists and overwrite is False. Skipping save.")
                return {
                    "ui": {"text": [f"⚠️ Skipped (Exists):\n{full_path}"]}, 
                    "result": (str(full_path),)
                }

        # 4. Tensor Conversion (Batch Processing)
        if image.shape[0] > 1:
            logger.warning("Batch size > 1 detected. Only saving the first image in batch.")
        
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 5. Format-Specific Saving Logic
        save_kwargs = {}
        
        if extension == "png":
            # Map quality 100->1 to compress_level 0->9
            compression = max(0, min(9, int((100 - quality) / 10)))
            save_kwargs["compress_level"] = compression
            save_kwargs["optimize"] = True
            
        elif extension == "jpg":
            if lossless:
                logger.warning("JPG does not support lossless mode. Ignoring toggle.")
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True

        elif extension == "webp":
            if lossless:
                save_kwargs["lossless"] = True
                save_kwargs["quality"] = 100
            else:
                save_kwargs["quality"] = quality
                save_kwargs["lossless"] = False

        # 6. Write
        try:
            img.save(str(full_path), **save_kwargs)
            logger.info(f"Saved image to: {full_path}")
        except Exception as e:
            raise IOError(f"Failed to save image: {e}")

        # 7. Return with UI Preview
        return {"ui": {"text": [f"Saved Image to:\n{full_path}"]}, "result": (str(full_path),)}


# --- Registration ---
NODE_CLASS_MAPPINGS = {
    "Qwen_TextView": Qwen_TextView,
    "Qwen_TextSave": Qwen_TextSave,
    "Qwen_ImageSave": Qwen_ImageSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen_TextView": "RM-Text Viewer",
    "Qwen_TextSave": "RM-Text Save",
    "Qwen_ImageSave": "RM-Image Save",
}