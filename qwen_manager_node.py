"""
Qwen3 Manager Node
Dedicated utility for managing (downloading/removing) Qwen3 models.
Separated into VL (Vision-Language) and Text-Only managers.

Optimized by: Principal Python Performance Engineer
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any

import folder_paths

# Try to import server for UI refresh, handle failure gracefully
try:
    import server
except ImportError:
    server = None

# --- Logging Setup ---
logger = logging.getLogger("Qwen3_Manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# --- Constants ---
# Directory for Vision Models (keep as VLM)
VLM_DIRECTORY = Path(folder_paths.models_dir) / "VLM"
VLM_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Directory for Text Models (typically LLM, or keep in VLM if preferred)
# Using 'LLM' to distinguish them in the filesystem
TEXT_DIRECTORY = Path(folder_paths.models_dir) / "LLM"
TEXT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Config Paths
JSON_VL_CONFIG = Path(__file__).parent / "qwen_vl_models.json"
JSON_TEXT_CONFIG = Path(__file__).parent / "qwen_text_models.json"


class Qwen3VL_ModelManager:
    """
    Manager for Vision-Language Models (Requires mmproj/projectors).
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return cls._get_input_types(JSON_VL_CONFIG, VLM_DIRECTORY)

    @staticmethod
    def _get_input_types(json_path: Path, model_dir: Path) -> Dict[str, Any]:
        raw_options: List[str] = []
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "huggingface_models" in data:
                    for section in data["huggingface_models"].values():
                        if isinstance(section, dict):
                            for model_info in section.values():
                                if isinstance(model_info, dict) and "repo_id" in model_info:
                                    raw_options.append(model_info["repo_id"])
                
                if "GGUF_models" in data:
                    gguf_root = data["GGUF_models"]
                    for key, val in gguf_root.items():
                        if isinstance(val, dict):
                            for m_info in val.values():
                                if isinstance(m_info, dict) and "model_files" in m_info:
                                    raw_options.extend(m_info["model_files"])
            except Exception as e:
                logger.error(f"Manager JSON Read Error ({json_path.name}): {e}")
        
        # UI Update: Removed the "*" logic. Just sort and display names.
        display_options = sorted(list(set(raw_options)))
        
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

    def _trigger_ui_refresh(self):
        if server:
            try:
                server.PromptServer.instance().send_sync("refresh_node_defs", {})
            except Exception:
                pass

    def manage_model(self, model_name: str, action: str) -> Tuple[str]:
        if model_name == "None Available":
            return ("Error: Config JSON not found.",)

        # Cleanup name (just in case)
        clean_name = model_name.strip()
        local_path = VLM_DIRECTORY / (clean_name.split("/")[-1] if "/" in clean_name else clean_name)
        is_gguf = clean_name.lower().endswith(".gguf")
        
        # --- Helper: Find GGUF details (Repo + Projector + Optional Projector Repo) ---
        def get_gguf_details(target: str) -> Tuple[str | None, str | None, str | None]:
            if not JSON_VL_CONFIG.exists(): return None, None, None
            try:
                with open(JSON_VL_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "GGUF_models" in data:
                    for cat_data in data["GGUF_models"].values():
                        if not isinstance(cat_data, dict): continue
                        for m_info in cat_data.values():
                            if isinstance(m_info, dict) and target in m_info.get("model_files", []):
                                return (
                                    m_info.get("repo_id"), 
                                    m_info.get("mmproj_file"),
                                    m_info.get("mmproj_repo_id")
                                )
            except Exception: pass
            return None, None, None

        # --- Helper: Check if Projector is used by OTHER installed models ---
        def is_projector_in_use(target_mmproj: str, current_model_name: str) -> bool:
            if not JSON_VL_CONFIG.exists(): return False
            try:
                with open(JSON_VL_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "GGUF_models" in data:
                    for cat_data in data["GGUF_models"].values():
                        if not isinstance(cat_data, dict): continue
                        for m_info in cat_data.values():
                            if not isinstance(m_info, dict): continue
                            
                            # Check models in this entry
                            files = m_info.get("model_files", [])
                            proj = m_info.get("mmproj_file")
                            
                            # If this entry uses the same projector
                            if proj == target_mmproj:
                                # Check every model file in this entry
                                for m_file in files:
                                    # Skip the model we are currently deleting
                                    if m_file == current_model_name:
                                        continue
                                    
                                    # Check if this OTHER model exists on disk
                                    other_path = VLM_DIRECTORY / m_file
                                    if other_path.exists():
                                        logger.info(f"🛡️ Projector protection: '{target_mmproj}' is still needed by '{m_file}'")
                                        return True
            except Exception as e:
                logger.warning(f"Error checking projector dependencies: {e}")
            return False

        # --- 1. CHECK STATUS ---
        if action == "Check Status":
            if local_path.exists():
                msg = f"✅ INSTALLED: {clean_name}"
                if is_gguf:
                    _, mmproj, _ = get_gguf_details(clean_name)
                    if mmproj:
                        if (VLM_DIRECTORY / mmproj).exists():
                            msg += f" (+ {mmproj})"
                        else:
                            msg += f" (⚠️ MISSING: {mmproj})"
                return (msg,)
            return (f"❌ MISSING: {clean_name}",)

        # --- 2. DOWNLOAD ---
        if action == "Download":
            try:
                from huggingface_hub import hf_hub_download, snapshot_download
                changes = False
                msg_parts = []

                if is_gguf:
                    repo, mmproj, mmproj_repo = get_gguf_details(clean_name)
                    if not repo: return (f"❌ Error: Config missing for {clean_name}",)
                    
                    # Download Main Model
                    if not local_path.exists():
                        logger.info(f"⬇️ Downloading GGUF: {clean_name}")
                        hf_hub_download(repo_id=repo, filename=clean_name, local_dir=str(VLM_DIRECTORY))
                        msg_parts.append(clean_name)
                        changes = True
                    else:
                        msg_parts.append(f"{clean_name} (Ready)")

                    # Download Projector
                    if mmproj:
                        mm_path = VLM_DIRECTORY / mmproj
                        if not mm_path.exists():
                            # Use mmproj_repo if defined, otherwise default to main repo
                            proj_download_repo = mmproj_repo if mmproj_repo else repo
                            
                            logger.info(f"⬇️ Downloading Projector: {mmproj} from {proj_download_repo}")
                            hf_hub_download(repo_id=proj_download_repo, filename=mmproj, local_dir=str(VLM_DIRECTORY))
                            msg_parts.append(f"& {mmproj}")
                            changes = True
                        else:
                            msg_parts.append(f"& {mmproj} (Ready)")
                    
                    if changes: self._trigger_ui_refresh()
                    return (f"✅ {' '.join(msg_parts)}",)

                else: # HF Repo
                    logger.info(f"⬇️ Downloading HF Repo: {clean_name}")
                    snapshot_download(repo_id=clean_name, local_dir=str(local_path))
                    self._trigger_ui_refresh()
                    return (f"✅ Downloaded {clean_name}",)

            except Exception as e:
                return (f"❌ Failed: {str(e)}",)

        # --- 3. REMOVE ---
        if action == "Remove":
            if not local_path.exists(): return ("ℹ️ Not found.",)
            try:
                if local_path.is_file(): local_path.unlink()
                elif local_path.is_dir(): shutil.rmtree(local_path)
                
                removed = [clean_name]
                
                # Smart Projector Removal
                if is_gguf:
                    _, mmproj, _ = get_gguf_details(clean_name)
                    if mmproj:
                        mm_path = VLM_DIRECTORY / mmproj
                        if mm_path.exists():
                            # CHECK: Is this projector used by any OTHER installed model?
                            if not is_projector_in_use(mmproj, clean_name):
                                mm_path.unlink()
                                removed.append(mmproj)
                            else:
                                removed.append(f"(kept {mmproj})")
                
                self._trigger_ui_refresh()
                return (f"🗑️ Deleted: {', '.join(removed)}",)
            except Exception as e:
                return (f"❌ Error: {str(e)}",)

        return ("Unknown Action",)


class Qwen3_TextModelManager:
    """
    Manager for Text-Only Models (LLMs).
    Does NOT handle mmproj files.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return Qwen3VL_ModelManager._get_input_types(JSON_TEXT_CONFIG, TEXT_DIRECTORY)

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "manage_model"
    CATEGORY = "Qwen3-VL"
    OUTPUT_NODE = True

    def _trigger_ui_refresh(self):
        if server:
            try:
                server.PromptServer.instance().send_sync("refresh_node_defs", {})
            except Exception: pass

    def manage_model(self, model_name: str, action: str) -> Tuple[str]:
        if model_name == "None Available":
            return ("Error: Config JSON not found.",)

        clean_name = model_name.strip()
        local_path = TEXT_DIRECTORY / (clean_name.split("/")[-1] if "/" in clean_name else clean_name)
        is_gguf = clean_name.lower().endswith(".gguf")
        
        def get_gguf_repo(target: str) -> str | None:
            if not JSON_TEXT_CONFIG.exists(): return None
            try:
                with open(JSON_TEXT_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "GGUF_models" in data:
                    for cat_data in data["GGUF_models"].values():
                        if not isinstance(cat_data, dict): continue
                        for m_info in cat_data.values():
                            if isinstance(m_info, dict) and target in m_info.get("model_files", []):
                                return m_info.get("repo_id")
            except Exception: pass
            return None

        if action == "Check Status":
            if local_path.exists():
                return (f"✅ INSTALLED: {clean_name}",)
            return (f"❌ MISSING: {clean_name}",)

        if action == "Download":
            if local_path.exists():
                return (f"ℹ️ ALREADY INSTALLED: {clean_name}",)
            try:
                from huggingface_hub import hf_hub_download, snapshot_download

                if is_gguf:
                    repo = get_gguf_repo(clean_name)
                    if not repo: return (f"❌ Error: Config missing for {clean_name}",)
                    
                    logger.info(f"⬇️ Downloading Text GGUF: {clean_name}")
                    hf_hub_download(repo_id=repo, filename=clean_name, local_dir=str(TEXT_DIRECTORY))
                    self._trigger_ui_refresh()
                    return (f"✅ Downloaded {clean_name}",)
                else:
                    logger.info(f"⬇️ Downloading Text HF Repo: {clean_name}")
                    snapshot_download(repo_id=clean_name, local_dir=str(local_path))
                    self._trigger_ui_refresh()
                    return (f"✅ Downloaded {clean_name}",)
            except Exception as e:
                return (f"❌ Failed: {str(e)}",)

        if action == "Remove":
            if not local_path.exists(): return ("ℹ️ Not found.",)
            try:
                if local_path.is_file(): local_path.unlink()
                elif local_path.is_dir(): shutil.rmtree(local_path)
                
                self._trigger_ui_refresh()
                return (f"🗑️ Deleted: {clean_name}",)
            except Exception as e:
                return (f"❌ Error: {str(e)}",)

        return ("Unknown Action",)


# Node Export mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ModelManager": Qwen3VL_ModelManager,
    "Qwen3_TextModelManager": Qwen3_TextModelManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ModelManager": "RM-Qwen3-VL Model Manager",
    "Qwen3_TextModelManager": "RM-Qwen3 Text Model Manager"
}