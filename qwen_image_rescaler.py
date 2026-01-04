import math
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, Tuple

# --- CONFIGURATION (Inherited from Loader for Consistency) ---
RES_PRESETS = tuple(str(i) for i in range(128, 2049, 64))
MOD_PRESETS = ("1", "2", "4", "8", "16", "32", "64")
RESAMPLING_METHODS = ["bicubic", "bilinear", "nearest", "box", "hamming", "lanczos"]

class Qwen3VL_ImageRescaler:
    """
    Standalone node for rescaling images using the same logic as the Qwen3-VL Universal Loader.
    Ensures consistent preprocessing for images not loaded via the loader node.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (RES_PRESETS, {"default": "1024"}),
                "modulo": (MOD_PRESETS, {"default": "4"}),
                "resampling": (RESAMPLING_METHODS,),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("resized_image", "width", "height")
    FUNCTION = "rescale_image"
    CATEGORY = "Qwen3-VL/Image"

    def rescale_image(self, image: torch.Tensor, resolution: str, modulo: str, resampling: str) -> Tuple[torch.Tensor, int, int]:
        # ComfyUI Image Tensor Shape: (Batch, Height, Width, Channels)
        
        out_images = []
        out_w, out_h = 0, 0
        
        for img_tensor in image:
            # 1. Convert Tensor [0..1] to Numpy uint8 [0..255] -> PIL
            np_img = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(np_img).convert("RGB")
            
            orig_w, orig_h = img.size
            
            # 2. Calculate Target Dimensions (Identical logic to Loader)
            res, mod = int(resolution), int(modulo)
            scale = math.sqrt((res**2) / (orig_w * orig_h))
            
            nw = max(mod, int(round(orig_w * scale / mod) * mod))
            nh = max(mod, int(round(orig_h * scale / mod) * mod))
            
            # 3. Resample
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
            
            # 4. Convert back to Tensor [0..1]
            resized_np = np.array(resized_img, dtype=np.float32) / 255.0
            out_images.append(torch.from_numpy(resized_np))
            
            out_w, out_h = nw, nh

        # Stack back into batch: (Batch, H, W, C)
        final_tensor = torch.stack(out_images)
        
        return (final_tensor, out_w, out_h)

# Add to mappings (Ensure you merge this dict with your main mappings)
NODE_CLASS_MAPPINGS = { "Qwen3VL_ImageRescaler": Qwen3VL_ImageRescaler }
NODE_DISPLAY_NAME_MAPPINGS = { "Qwen3VL_ImageRescaler": "RM-Qwen3-VL Image Rescaler" }