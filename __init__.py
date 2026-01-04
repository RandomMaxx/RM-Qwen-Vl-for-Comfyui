"""
Qwen3-VL Node Pack Registration
Optimized by: Principal Python Performance Engineer
"""

from .qwen_image_node import NODE_CLASS_MAPPINGS as IMAGE_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_NAMES
from .qwen_image_rescaler import NODE_CLASS_MAPPINGS as RESCALER_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as RESCALER_NAMES
from .qwen_3_vl_comfyui_node import NODE_CLASS_MAPPINGS as MODEL_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as MODEL_NAMES
from .qwen_manager_node import NODE_CLASS_MAPPINGS as MANAGER_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as MANAGER_NAMES

# Merge all mappings into the final exports
NODE_CLASS_MAPPINGS = {
    **IMAGE_CLASSES,
    **RESCALER_CLASSES,
    **MODEL_CLASSES,
    **MANAGER_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **IMAGE_NAMES,
    **RESCALER_NAMES,
    **MODEL_NAMES,
    **MANAGER_NAMES,
}

# Registers the 'js' folder for frontend extensions (Javascript)
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]