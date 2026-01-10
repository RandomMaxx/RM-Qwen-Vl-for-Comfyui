import math
import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, Tuple, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# Shared constants for UI inputs
RES_PRESETS = tuple(str(i) for i in range(128, 2049, 64))
MOD_PRESETS = ("1", "2", "4", "8", "16", "32", "64")
# Full PIL Resampling List ordered by common utility
RESAMPLING_METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "box", "hamming"]

class Qwen3VL_ImageRescaler:
    """
    Standard Rescaler Node with optional Sharpening.

    Resizes images based on total pixel area to match a target resolution 
    while maintaining aspect ratio, then snaps dimensions to a modulo.
    Does NOT crop.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (RES_PRESETS, {"default": "1024"}),
                "modulo": (MOD_PRESETS, {"default": "4"}),
                "resampling": (RESAMPLING_METHODS,),
                "enable_sharpening": ("BOOLEAN", {"default": False}),
                "sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sharpen_percent": ("INT", {"default": 50, "min": 0, "max": 300, "step": 1}),
                "sharpen_threshold": ("INT", {"default": 5, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "rescale_image"
    CATEGORY = "Qwen3-VL/Image"

    def rescale_image(
        self, 
        image: torch.Tensor, 
        resolution: str, 
        modulo: str, 
        resampling: str,
        enable_sharpening: bool,
        sharpen_radius: float, 
        sharpen_percent: int, 
        sharpen_threshold: int
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Rescales a batch of images using multi-threading.

        Args:
            image (torch.Tensor): Input batch (B, H, W, C).
            resolution (str): Target resolution basis.
            modulo (str): Divisibility constraint.
            resampling (str): Interpolation method.
            enable_sharpening (bool): Whether to apply unsharp mask.
            sharpen_radius (float): Radius of the sharpening kernel.
            sharpen_percent (int): Strength of sharpening.
            sharpen_threshold (int): Minimum brightness change to be sharpened.

        Returns:
            Tuple[torch.Tensor, int, int]: Resized batch, new width, new height.
        """
        res, mod = int(resolution), int(modulo)
        
        # Mapping string inputs to PIL constants
        resample_map = {
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
            "lanczos": Image.Resampling.LANCZOS
        }
        method = resample_map.get(resampling, Image.Resampling.BICUBIC)

        def _process_single(img_tensor: torch.Tensor) -> Tuple[np.ndarray, int, int]:
            # 1. Convert Tensor [0..1] to Numpy uint8 [0..255] -> PIL
            np_img = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(np_img).convert("RGB")
            
            # 2. Optional Sharpening
            if enable_sharpening and sharpen_percent > 0:
                img = img.filter(ImageFilter.UnsharpMask(
                    radius=sharpen_radius, 
                    percent=sharpen_percent, 
                    threshold=sharpen_threshold
                ))

            orig_w, orig_h = img.size
            
            # 3. Calculate Target Dimensions (Area based)
            scale = math.sqrt((res**2) / (orig_w * orig_h))
            
            nw = max(mod, int(round(orig_w * scale / mod) * mod))
            nh = max(mod, int(round(orig_h * scale / mod) * mod))
            
            # 4. Resample
            if nw != orig_w or nh != orig_h:
                img = img.resize((nw, nh), method)
            
            # 5. Convert back to float32 array
            return np.array(img, dtype=np.float32) / 255.0, nw, nh

        # Execute in ThreadPool
        if len(image) == 0:
             return (image, 0, 0)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_process_single, image))
            
        processed_arrays, widths, heights = zip(*results)
        
        # Stack back into batch: (Batch, H, W, C)
        final_tensor = torch.from_numpy(np.stack(processed_arrays))
        
        # Return dims of the first image
        return (final_tensor, widths[0], heights[0])


class Qwen3VL_ResizeCrop:
    """
    High-Performance Image Processor with Toggleable Crop/Sharpen.
    
    Features:
    - Multi-threaded Batch Processing.
    - Optional Unsharp Mask Sharpening.
    - Smart Aspect Ratio: Resizes to cover target.
    - Optional Center Crop to exact modulo dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (RES_PRESETS, {"default": "512"}),
                "modulo": (MOD_PRESETS, {"default": "32"}),
                "resampling": (RESAMPLING_METHODS, {"default": "lanczos"}),
                "enable_cropping": ("BOOLEAN", {"default": True}),
                "enable_sharpening": ("BOOLEAN", {"default": True}),
                "sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sharpen_percent": ("INT", {"default": 50, "min": 0, "max": 300, "step": 1}),
                "sharpen_threshold": ("INT", {"default": 5, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process_batch"
    CATEGORY = "Qwen3-VL/Image"

    def process_batch(
        self, 
        image: torch.Tensor, 
        resolution: str, 
        modulo: str, 
        resampling: str,
        enable_cropping: bool,
        enable_sharpening: bool,
        sharpen_radius: float, 
        sharpen_percent: int, 
        sharpen_threshold: int
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Executes the resize-crop-sharpen pipeline using a thread pool.
        """
        target_res = int(resolution)
        mod = int(modulo)
        
        # PIL Resampling Map
        resample_map = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        filter_method = resample_map.get(resampling, Image.Resampling.LANCZOS)

        def _process_single(img_tensor: torch.Tensor) -> Tuple[np.ndarray, int, int]:
            """
            Inner function to process a single image frame (Runs in Thread).
            """
            # 1. Tensor -> PIL
            np_img = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(np_img).convert("RGB")
            
            # 2. Optional Sharpening
            if enable_sharpening and sharpen_percent > 0:
                img = img.filter(ImageFilter.UnsharpMask(
                    radius=sharpen_radius, 
                    percent=sharpen_percent, 
                    threshold=sharpen_threshold
                ))

            # 3. Calculate Dimensions
            orig_w, orig_h = img.size
            
            # Logic: Long side = target_res (snapped), Short side = aspect (snapped)
            snapped_res = max(mod, (target_res // mod) * mod)
            
            if orig_w >= orig_h: # Landscape/Square
                target_w = snapped_res
                target_h = max(mod, int(round((target_w / orig_w) * orig_h / mod) * mod))
            else: # Portrait
                target_h = snapped_res
                target_w = max(mod, int(round((target_h / orig_h) * orig_w / mod) * mod))

            # 4. Resize to COVER (Ensure image >= target dimensions)
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            scale = max(scale_w, scale_h) 
            
            resize_w = int(round(orig_w * scale))
            resize_h = int(round(orig_h * scale))
            
            if resize_w != orig_w or resize_h != orig_h:
                img = img.resize((resize_w, resize_h), filter_method)
            
            # 5. Optional Center Crop
            final_w, final_h = resize_w, resize_h
            
            if enable_cropping:
                # If resizing resulted in dimensions larger than target, crop to target.
                if resize_w > target_w or resize_h > target_h:
                    left = (resize_w - target_w) // 2
                    top = (resize_h - target_h) // 2
                    img = img.crop((left, top, left + target_w, top + target_h))
                    final_w, final_h = target_w, target_h
            
            # 6. PIL -> Numpy (float32, 0..1)
            return np.array(img, dtype=np.float32) / 255.0, final_w, final_h

        if len(image) == 0:
            return (image, 0, 0)

        # Use ThreadPool to parallelize the batch processing.
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_process_single, image))
        
        processed_arrays, widths, heights = zip(*results)
        
        # Stack back to Tensor: (Batch, H, W, C)
        final_tensor = torch.from_numpy(np.stack(processed_arrays))
        
        # Return dims of the first image (Note: if cropping is disabled, dims might vary if batch input aspect ratios varied)
        # We return the first image's dims as standard ComfyUI behavior assumes consistent batch size.
        return (final_tensor, widths[0], heights[0])


# --- NODE MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ImageRescaler": Qwen3VL_ImageRescaler,
    "Qwen3VL_ResizeCrop": Qwen3VL_ResizeCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ImageRescaler": "RM-Qwen3-VL Image Rescaler",
    "Qwen3VL_ResizeCrop": "RM-Qwen3-VL Resize & Crop"
}