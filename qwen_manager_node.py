"""
Qwen3-VL Manager Node
Dedicated utility for managing (downloading/removing) Qwen3-VL models.
Isolated from the main inference pipeline for stability and clean separation.

Optimized by: Principal Python Performance Engineer
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any

import folder_paths

# --- Logging Setup (Consistent with Main Node) ---
logger = logging.getLogger("Qwen3VL_Manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# --- Constants ---
# Ensures we target the exact same directory as the Loader node
MODEL_DIRECTORY = Path(folder_paths.models_dir) / "VLM"
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Assumes JSON is in the same directory as this script
JSON_CONFIG_PATH = Path(__file__).parent / "qwen_models.json"


class Qwen3VL_ModelManager:
    """
    Dedicated node for downloading, checking, and removing models.
    Does not load models into memory. Safe for low-VRAM management.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Dynamic input generation. Scans the JSON for available models
        and checks the filesystem to mark uninstalled models with a '*'.
        """
        raw_options: List[str] = []
        
        # 1. Parse JSON for raw model names
        if JSON_CONFIG_PATH.exists():
            try:
                with open(JSON_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # HF Models (Repo IDs)
                if "huggingface_models" in data:
                    hf_section = data["huggingface_models"]
                    for category in hf_section.values():
                        if isinstance(category, dict):
                            for model_info in category.values():
                                if isinstance(model_info, dict) and "repo_id" in model_info:
                                    raw_options.append(model_info["repo_id"])
                
                # GGUF Models (Filenames)
                if "GGUF_models" in data:
                    gguf_section = data["GGUF_models"]
                    for category in gguf_section.values():
                        if isinstance(category, dict):
                            for model_info in category.values():
                                if isinstance(model_info, dict) and "model_files" in model_info:
                                    raw_options.extend(model_info["model_files"])
            except Exception as e:
                logger.error(f"Manager JSON Read Error: {e}")
        
        raw_options = sorted(list(set(raw_options)))
        
        # 2. Check installation status and format display names
        display_options: List[str] = []
        
        for name in raw_options:
            # Determine expected local path logic (must match manage_model logic)
            if "/" in name:
                # It's a Repo ID, it creates a folder named after the last segment
                local_path = MODEL_DIRECTORY / name.split("/")[-1]
            else:
                # It's a file (GGUF)
                local_path = MODEL_DIRECTORY / name
            
            # If not installed, prepend asterisk
            if not local_path.exists():
                display_options.append(f"* {name}")
            else:
                display_options.append(name)

        if not display_options:
            display_options = ["None Available"]

        return {
            "required": {
                "model_name": (display_options,),
                "action": (["Check Status", "Download", "Remove"], {"default": "Check Status"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "manage_model"
    CATEGORY = "Qwen3-VL"
    OUTPUT_NODE = True

    def manage_model(self, model_name: str, action: str) -> Tuple[str]:
        """
        Execute the management action.
        
        Args:
            model_name (str): The name selected in the UI (may contain '* ').
            action (str): The action to perform.

        Returns:
            Tuple[str]: Status message.
        """
        if model_name == "None Available":
            return ("Error: qwen_models.json not found or empty.",)

        # --- SANITIZE INPUT ---
        # Strip the "* " indicator used in the UI to identify uninstalled models
        clean_name = model_name.lstrip("* ").strip()
            
        # Determine local path based on clean name
        if "/" in clean_name:
             local_path = MODEL_DIRECTORY / clean_name.split("/")[-1]
        else:
             local_path = MODEL_DIRECTORY / clean_name
             
        is_gguf = clean_name.lower().endswith(".gguf")
        
        # --- 1. CHECK STATUS ---
        if action == "Check Status":
            if local_path.exists():
                return (f"✅ INSTALLED: {clean_name} is present at {local_path}",)
            return (f"❌ MISSING: {clean_name} is not installed.",)

        # --- 2. DOWNLOAD ---
        if action == "Download":
            if local_path.exists():
                return (f"ℹ️ ALREADY EXISTS: {clean_name} is already installed.",)
            
            try:
                from huggingface_hub import hf_hub_download, snapshot_download
                
                if is_gguf:
                    # GGUF Logic: We must find the repo_id and mmproj file from the JSON
                    target_repo = None
                    mmproj_target = None
                    
                    if not JSON_CONFIG_PATH.exists():
                        return ("❌ ERROR: qwen_models.json missing, cannot look up download details.",)

                    with open(JSON_CONFIG_PATH, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Search for the filename in the JSON structure
                    found = False
                    if "GGUF_models" in data:
                        for cat in data["GGUF_models"].values():
                            if not isinstance(cat, dict): continue
                            for m_info in cat.values():
                                if not isinstance(m_info, dict): continue
                                # Check against clean_name
                                if clean_name in m_info.get("model_files", []):
                                    target_repo = m_info.get("repo_id")
                                    mmproj_target = m_info.get("mmproj_file")
                                    found = True
                                    break
                            if found: break
                    
                    if not target_repo:
                        return (f"❌ ERROR: Could not find repo configuration for '{clean_name}' in JSON.",)
                        
                    logger.info(f"⬇️ Downloading GGUF: {clean_name} from {target_repo}")
                    hf_hub_download(repo_id=target_repo, filename=clean_name, local_dir=str(MODEL_DIRECTORY))
                    
                    if mmproj_target:
                        mm_path = MODEL_DIRECTORY / mmproj_target
                        if not mm_path.exists():
                            logger.info(f"⬇️ Downloading Projector: {mmproj_target}")
                            hf_hub_download(repo_id=target_repo, filename=mmproj_target, local_dir=str(MODEL_DIRECTORY))
                    
                    return (f"✅ DOWNLOAD COMPLETE: {clean_name} (and required projector)",)

                else:
                    # Transformers Folder Logic
                    logger.info(f"⬇️ Downloading HF Repo: {clean_name}")
                    # local_dir will be 'models/VLM/RepoName'
                    target_dir = MODEL_DIRECTORY / clean_name.split("/")[-1]
                    snapshot_download(repo_id=clean_name, local_dir=str(target_dir))
                    return (f"✅ DOWNLOAD COMPLETE: {clean_name} folder created.",)

            except Exception as e:
                logger.error(f"Download Failed: {e}")
                return (f"❌ DOWNLOAD FAILED: {str(e)}",)

        # --- 3. REMOVE ---
        if action == "Remove":
            if not local_path.exists():
                 return (f"ℹ️ NOTHING TO REMOVE: {clean_name} not found.",)
            
            try:
                if local_path.is_file():
                    local_path.unlink() # Delete file
                    return (f"🗑️ DELETED: {clean_name} file removed.",)
                elif local_path.is_dir():
                    shutil.rmtree(local_path)
                    return (f"🗑️ DELETED: {clean_name} folder removed.",)
            except Exception as e:
                 return (f"❌ DELETION FAILED: {str(e)}",)

        return ("Unknown Action",)

# Node Export mappings for this file
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ModelManager": Qwen3VL_ModelManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ModelManager": "RM-Qwen3-VL Model Manager"
}