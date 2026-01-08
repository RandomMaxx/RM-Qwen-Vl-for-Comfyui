"""
Qwen3-VL ComfyUI Custom Node - High Performance Edition
Unified node for Vision-Language tasks with support for all Qwen3-VL models.
Supports GGUF (llama.cpp) and Transformers (HuggingFace) backends.
Includes Advanced Generation Configuration.

Optimized by: Principal Python Performance Engineer
Status: Production Grade (In-Memory Processing, Local Only)
"""

import gc
import logging
import re
import base64
import os
import io
import json
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

# --- Logging Setup ---
logger = logging.getLogger("Qwen3VL")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# --- Dependency Guard ---
# 1. Transformers (Standard)
try:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    process_vision_info = None
    Qwen3VLForConditionalGeneration = None

# 2. Llama-CPP (GGUF)
try:
    import llama_cpp
    from llama_cpp import Llama
    
    # Log version for debugging
    logger.info(f"Using llama-cpp-python version: {llama_cpp.__version__}")

    # Attempt to import specific Qwen handlers
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler as QwenChatHandler
    except ImportError:
        try:
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler as QwenChatHandler
        except ImportError:
            QwenChatHandler = None
            logger.warning("Could not import Qwen2/2.5 VL ChatHandler. GGUF loading may fail for VL models.")
            
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    QwenChatHandler = None


# --- Constants ---
MODEL_DIRECTORY = Path(folder_paths.models_dir) / "VLM"
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
JSON_CONFIG_PATH = Path(__file__).parent / "qwen_models.json"


# --- Configuration Maps ---
CAPTION_TYPE_MAP = {
    "Descriptive (Standard)": [
        "Describe this image in detail, covering the subject, background, and lighting.",
        "Describe the image in exactly {word_count} words.",
        "Write a {length} description of the image.",
    ],
    "Straightforward (Concise)": [
        "Describe the main subject and action only. No flowery language.",
        "Concise description in under {word_count} words.",
        "Concise {length} description.",
    ],
    "Z-Image Turbo (Structured)": [
        "Analyze the image and generate an image description in this strict order: [Subject], [Action/Context], [Lighting], [Camera/Film Spec], [Positive Constraints (Sharp, 8k)].",
        "Generate an image description in {word_count} words, focusing on subject, style, composition, camera and lighting.",
        "Write a {length} structured image description with emphasized subject, style and lighting details.",
    ],
    "Flux.1 (Natural Narrative)": [
        "Describe the image using rich, natural language. Focus on textures, atmosphere, and small details. Avoid list-style formatting.",
        "Write a detailed narrative image description in {word_count} words.",
        "Write a {length} natural language description focusing on atmosphere.",
    ],
    "Wan 2.1 / Video (SSM Formula)": [
        "Describe the video potential of this image using the SSM formula: [Subject Description] + [Scene/Background] + [Implied Motion/Action].",
        "Write a video generation prompt in {word_count} words focusing on movement.",
        "Write a {length} video prompt describing the subject and the specific camera movement.",
    ],
    "Danbooru Tags (Training)": [
        "Generate a list of Danbooru-style tags, separated by commas. Include character tags, clothing, and background.",
        "Generate approximately {word_count} Danbooru tags.",
        "Generate a {length} list of Danbooru tags.",
    ],
    "Photography Inspection": [
        "Analyze the technical photography aspects: Estimate the camera angle, focal length, aperture (depth of field), and lighting setup.",
        "Technical photography analysis in {word_count} words.",
        "Write a {length} technical report on the camera settings used.",
    ],
    "OCR / Text Extraction": [
        "Transcribe all visible text in the image exactly as it appears. Maintain line breaks if possible.",
        "Extract text in {word_count} words.",
        "Extract text, {length} output.",
    ],
}

SHARED_BOOLEAN_OPTIONS = {
    "Identify art style": ("BOOLEAN", {"default": False, "label_on": "Identify Art Style/Medium"}),
    "Describe mood": ("BOOLEAN", {"default": False, "label_on": "Describe Mood/Atmosphere"}),
    "Focus on main subject": ("BOOLEAN", {"default": True, "label_on": "Focus on Main Subject"}),
    "Describe clothing": ("BOOLEAN", {"default": False, "label_on": "Describe Clothing"}),
    "Describe background": ("BOOLEAN", {"default": False, "label_on": "Describe Background"}),
    "Transcribe text": ("BOOLEAN", {"default": False, "label_on": "Read/Transcribe Text"}),
    "Mention watermarks": ("BOOLEAN", {"default": False, "label_on": "Mention watermarks"}),
    "Analyze lighting": ("BOOLEAN", {"default": False, "label_on": "Include lighting info"}),
    "Analyze camera angle": ("BOOLEAN", {"default": False, "label_on": "Include camera angle"}),
    "Analyze composition": ("BOOLEAN", {"default": False, "label_on": "Describe composition"}),
    "Analyze depth": ("BOOLEAN", {"default": False, "label_on": "Describe depth/focus"}),
    "Rate quality": ("BOOLEAN", {"default": False, "label_on": "Rate Technical Quality"}),
    "Be concise": ("BOOLEAN", {"default": False, "label_on": "Be Concise/Brief"}),
    "Omit artist names": ("BOOLEAN", {"default": False, "label_on": "Don't mention artist names"}),
    "Omit text detection": ("BOOLEAN", {"default": False, "label_on": "Don't mention text detection"}),
}

OPTION_TEXT_MAP = {
    "Identify art style": "Identify the art style and medium used.",
    "Describe mood": "Describe the overall mood, atmosphere, and emotional tone.",
    "Focus on main subject": "Focus primarily on describing the main subject.",
    "Describe clothing": "Describe the clothing, accessories, and fashion style in detail.",
    "Describe background": "Provide a description of the background, setting, and environment.",
    "Transcribe text": "Transcribe any visible text found within the image.",
    "Mention watermarks": "Note if there are any watermarks, signatures, or logos visible.",
    "Analyze lighting": "Include detailed information about the lighting setup.",
    "Analyze camera angle": "Describe the camera angle and shot type.",
    "Analyze composition": "Analyze the visual structure, subject placement, and use of spatial guides (such as the Rule of Thirds or Golden Ratio).",
    "Analyze depth": "Describe the depth of field, focus, and blur.",
    "Rate quality": "Assess technical aspects like sharpness, focus, and noise levels.",
    "Be concise": "Keep the description concise and to the point.",
    "Omit artist names": "Do NOT guess or mention specific artist names.",
    "Omit text detection": "Do NOT mention the presence or absence of text in the image.",
}



def clean_vram(model_dict: Optional[Dict[str, Any]] = None, unload: bool = False) -> None:
    """Performs unified VRAM cleanup and optional model offloading."""
    if unload and model_dict:
        model = model_dict.get("model")
        backend = model_dict.get("backend", "transformers")

        if backend == "transformers" and model:
            try:
                model.to("cpu")
            except Exception:
                pass 
        elif backend == "gguf" and model:
            try:
                if hasattr(model, "close"):
                    model.close()
                elif hasattr(model, "_model") and hasattr(model._model, "close"):
                    model._model.close()
                keys = list(model_dict.keys())
                for k in keys:
                    del model_dict[k]
                del model
            except Exception as e:
                logger.debug(f"GGUF cleanup warning: {e}")

    mm.soft_empty_cache()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


class Qwen3VL_GenerationConfig:
    """
    Advanced configuration node for controlling model generation stochasticity.
    Connects between Model Loader and Run Node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Higher = Creative, Lower = Deterministic"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repeating tokens"}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Beam search (Transformers Only). 1 = Standard"}),
            }
        }

    RETURN_TYPES = ("QWEN3_GEN_CONFIG",)
    RETURN_NAMES = ("gen_config",)
    FUNCTION = "create_config"
    CATEGORY = "Qwen3-VL"

    def create_config(self, temperature: float, top_p: float, repetition_penalty: float, num_beams: int) -> Tuple[Dict[str, Any]]:
        """Packages generation parameters into a dictionary."""
        return ({
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
        },)


class Qwen3VL_Base:
    """Base class containing shared logic and strictly in-memory I/O handling."""

    CATEGORY = "Qwen3-VL"

    @classmethod
    def get_shared_options(cls) -> Dict:
        return SHARED_BOOLEAN_OPTIONS

    def build_prompt_text(
        self, 
        caption_type: str, 
        caption_length: str, 
        custom_prompt: str, 
        prompt_mode: str,
        kwargs: Dict[str, Any]
    ) -> str:
        # 1. Handle Overwrite immediately
        if prompt_mode == "overwrite" and custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()

        # 2. Construct Template Prompt
        templates = CAPTION_TYPE_MAP.get(caption_type, CAPTION_TYPE_MAP["Descriptive (Standard)"])

        if caption_length == "any":
            base_prompt = templates[0]
        elif caption_length.isdigit():
            base_prompt = templates[1].format(word_count=caption_length)
        else:
            base_prompt = templates[2].format(length=caption_length)

        # 3. Add Instructions
        active_instructions = [
            f"- {OPTION_TEXT_MAP[k]}"
            for k, v in kwargs.items()
            if v is True and k in OPTION_TEXT_MAP
        ]
        
        if active_instructions:
            base_prompt += "\n\nAdditional Instructions:\n" + "\n".join(active_instructions)

        # 4. Handle Concatenation (Prepend)
        if prompt_mode == "prepend_custom" and custom_prompt and custom_prompt.strip():
            return f"{custom_prompt.strip()}\n\n{base_prompt}"

        return base_prompt

    def _tensor_to_base64_uri(self, image_tensor: torch.Tensor) -> str:
        if image_tensor.dim() != 3:
             raise ValueError(f"Expected [H, W, C] tensor, got shape {image_tensor.shape}")
        
        arr = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
            
        img = Image.fromarray(arr)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", compress_level=1)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def run_inference_transformers(
        self, model_dict: Dict, messages: List[Dict], max_new_tokens: int, gen_config: Optional[Dict] = None
    ) -> str:
        """Inference logic for Transformers backend with dynamic generation config."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available.")
            
        model = model_dict["model"]
        processor = model_dict["processor"]

        # Default Config if None provided
        if gen_config is None:
            gen_config = {"temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.2, "num_beams": 1}

        # Prepare arguments
        do_sample = gen_config["temperature"] > 0
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": gen_config["repetition_penalty"],
            "num_beams": gen_config["num_beams"],
            "use_cache": True,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = gen_config["temperature"]
            gen_kwargs["top_p"] = gen_config["top_p"]

        try:
            device = next(model.parameters()).device
            if str(device) == "cpu" and torch.cuda.is_available():
                model.to("cuda")
        except Exception:
            pass

        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            
            result = str(output_text[0])
            if "</think>" in result:
                result = result.split("</think>")[-1]
                
            return re.sub(r"^[\s\u200b\xa0]+", "", result)

        except torch.cuda.OutOfMemoryError as e:
            clean_vram(model_dict, unload=True)
            raise RuntimeError("CUDA Out of Memory. Model unloaded.") from e
        except Exception as e:
            raise RuntimeError(f"Transformers Inference failed: {str(e)}") from e

    def run_inference_gguf(
        self, model_dict: Dict, messages: List[Dict], max_new_tokens: int, gen_config: Optional[Dict] = None
    ) -> str:
        """Inference logic for GGUF backend with dynamic generation config."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available.")

        model = model_dict["model"]
        
        # Default Config
        if gen_config is None:
            gen_config = {"temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.2, "num_beams": 1}

        # GGUF Mapping
        inference_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repeat_penalty": gen_config["repetition_penalty"],
        }

        openai_messages = []
        for msg in messages:
            role = msg["role"]
            content_list = msg["content"]
            
            if isinstance(content_list, str):
                openai_messages.append({"role": role, "content": content_list})
                continue
            
            new_content = []
            for item in content_list:
                if item["type"] == "text":
                    new_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    new_content.append({
                        "type": "image_url", 
                        "image_url": {"url": item["image"]}
                    })
            
            openai_messages.append({"role": role, "content": new_content})

        try:
            response = model.create_chat_completion(
                messages=openai_messages,
                **inference_kwargs
            )
            
            result = response["choices"][0]["message"]["content"]
            if "</think>" in result:
                result = result.split("</think>")[-1]

            return re.sub(r"^[\s\u200b\xa0]+", "", result)
            
        except Exception as e:
            raise RuntimeError(f"GGUF Inference failed: {str(e)}") from e


class Qwen3VL_ModelLoader:
    """Loads Qwen3-VL models (HF + GGUF)."""

    @classmethod
    def INPUT_TYPES(cls):
        model_options = set()
        if MODEL_DIRECTORY.exists():
            for item in MODEL_DIRECTORY.iterdir():
                if item.is_dir() and item.name != ".cache":
                    model_options.add(item.name)
                elif item.is_file() and item.suffix.lower() == ".gguf" and "mmproj" not in item.name.lower():
                    model_options.add(item.name)

        sorted_options = sorted(list(model_options))
        if not sorted_options:
             sorted_options = ["No models found in VLM folder"]

        return {
            "required": {
                "model": (sorted_options, {"default": sorted_options[0] if sorted_options else ""}),
                "quantization": (["none", "4bit", "8bit"], {"default": "8bit"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
                "gpu_layers_gguf": ("INT", {"default": -1, "min": -1, "max": 128, "tooltip": "GGUF Only: -1 = all layers"}),
            },
        }

    RETURN_TYPES = ("QWEN3_VL_MODEL",)
    RETURN_NAMES = ("model_dict",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-VL"

    def _find_mmproj(self, model_path: Path) -> Optional[str]:
        parent = model_path.parent
        candidates = [p for p in parent.glob("*mmproj*.gguf") if p.is_file()]
        
        if not candidates:
            return None

        model_name = model_path.name.lower()
        best_candidate = None
        best_ratio = 0.0

        for cand in candidates:
            cand_name = cand.name.lower()
            ratio = difflib.SequenceMatcher(None, model_name, cand_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = cand

        if best_candidate and best_ratio > 0.3:
            return str(best_candidate)
        return None
    
    def load_model(self, model: str, quantization: str, attention: str, gpu_layers_gguf: int):
        if model == "No models found in VLM folder":
            raise FileNotFoundError("No models found in 'models/VLM'.")

        if "/" in model:
             return self.load_model_transformers(model, quantization, attention)
        
        local_path = MODEL_DIRECTORY / model
        is_gguf = model.lower().endswith(".gguf")
        
        if is_gguf:
            if local_path.exists():
                return self.load_model_gguf(local_path, gpu_layers_gguf)
            else:
                 raise FileNotFoundError(f"GGUF file '{model}' not found locally.")
            
        return self.load_model_transformers(model, quantization, attention)

    def load_model_gguf(self, model_path: Path, gpu_layers: int):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("Cannot load GGUF. 'llama-cpp-python' is not installed.")

        mmproj_path = self._find_mmproj(model_path)
        if not mmproj_path:
            raise FileNotFoundError(f"Missing vision projector (mmproj) for {model_path.name}")
        
        if QwenChatHandler is None:
             raise ImportError("Installed 'llama-cpp-python' does not support Qwen-VL ChatHandler.")
        
        try:
            chat_handler = QwenChatHandler(clip_model_path=mmproj_path)
        except TypeError as e:
             raise RuntimeError(f"ChatHandler Init Error: {e}") from e

        logger.info(f"🔧 Loading GGUF Model: {model_path.name}...")
        try:
            llm = Llama(
                model_path=str(model_path),
                chat_handler=chat_handler,
                n_gpu_layers=gpu_layers,
                n_ctx=4096,
                verbose=True
            )
        except Exception as e:
            raise RuntimeError(f"GGUF Load Error: {e}") from e

        return ({"model": llm, "type": "gguf", "model_path": str(model_path), "backend": "gguf"},)

    def load_model_transformers(self, model: str, quantization: str, attention: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not installed.")

        if "/" in model:
            model_name = model.rsplit("/", 1)[-1]
        else:
            model_name = model
        model_path = MODEL_DIRECTORY / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Local model folder '{model_name}' not found.")

        if attention == "auto":
            try:
                from transformers.utils import is_flash_attn_2_available
                if not is_flash_attn_2_available():
                    attention = "sdpa"
            except Exception:
                attention = "sdpa"

        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"🔧 Loading Transformers Model: {model_name}...")
        try:
            loaded_model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quant_config,
            )
            processor = AutoProcessor.from_pretrained(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Transformers Load Error: {e}") from e

        return ({"model": loaded_model, "processor": processor, "model_path": str(model_path), "type": "transformers", "backend": "transformers"},)


class Qwen3VL_Run(Qwen3VL_Base):
    """Single Image/Video Inference Node with optional Advanced Config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dict": ("QWEN3_VL_MODEL",),
                "caption_type": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive (Standard)"}),
                "caption_length": (
                    ["any", "short", "long"] + [str(i) for i in range(12, 512, 12)],
                    {"default": "long"},
                ),
                "custom_prompt": ("STRING", {"multiline": True}),
                "prompt_mode": (["prepend_custom", "overwrite"], {"default": "prepend_custom"}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a masterful assistant and you describe images in natural language. Write the descriptions in one fluid paragraph.", 
                        "multiline": True
                    },
                ),
                "initial_tags": ("STRING", {"multiline": True, "default": ""}),
                "end_tags": ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 1}),
                "unload_when_done": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "generation_config": ("QWEN3_GEN_CONFIG",),
                **cls.get_shared_options(),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output", "user_prompt", "system_prompt")
    FUNCTION = "run"

    def run(
        self,
        model_dict,
        caption_type,
        caption_length,
        custom_prompt,
        prompt_mode,
        system_prompt,
        initial_tags,
        end_tags,
        max_new_tokens,
        seed,
        unload_when_done,
        image=None,
        generation_config=None,
        **kwargs,
    ):
        user_prompt = self.build_prompt_text(
            caption_type, caption_length, custom_prompt, prompt_mode, kwargs
        )
        
        try:
            content = []
            if image is not None:
                batch_size = image.shape[0]
                for i in range(batch_size):
                    b64_uri = self._tensor_to_base64_uri(image[i])
                    content.append({"type": "image", "image": b64_uri})

            content.append({"type": "text", "text": user_prompt})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]

            backend = model_dict.get("backend", "transformers")
            
            if backend == "gguf":
                result = self.run_inference_gguf(model_dict, messages, max_new_tokens, generation_config)
            else:
                result = self.run_inference_transformers(model_dict, messages, max_new_tokens, generation_config)

            parts = []
            if initial_tags and initial_tags.strip():
                parts.append(initial_tags.strip())
            parts.append(result)
            if end_tags and end_tags.strip():
                parts.append(end_tags.strip())
            
            final_output = " ".join(parts)
            return (final_output, user_prompt, system_prompt)

        finally:
            clean_vram(model_dict, unload=unload_when_done)

class Qwen3VL_Run_Simple(Qwen3VL_Base):
    """Simplified Inference Node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dict": ("QWEN3_VL_MODEL",),
                "image": ("IMAGE",),
                "caption_type": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive (Standard)"}),
                "caption_length": (
                    ["any", "short", "long"] + [str(i) for i in range(12, 512, 12)],
                    {"default": "long"},
                ),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_mode": (["prepend_custom", "overwrite"], {"default": "prepend_custom"}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a masterful assistant and you describe images in natural language. Write the descriptions in one fluid paragraph.",
                        "multiline": True
                    },
                ),
                "initial_tags": ("STRING", {"multiline": True, "default": ""}),
                "end_tags": ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 1}),
                "unload_when_done": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "generation_config": ("QWEN3_GEN_CONFIG",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output", "user_prompt", "system_prompt")
    FUNCTION = "run"

    def run(
        self,
        model_dict,
        image,
        caption_type,
        caption_length,
        custom_prompt,
        prompt_mode,
        system_prompt,
        initial_tags,
        end_tags,
        max_new_tokens,
        seed,
        unload_when_done,
        generation_config=None,
    ):
        user_prompt = self.build_prompt_text(
            caption_type, caption_length, custom_prompt, prompt_mode, {}
        )
        
        try:
            content = []
            if image is not None:
                batch_size = image.shape[0]
                for i in range(batch_size):
                    b64_uri = self._tensor_to_base64_uri(image[i])
                    content.append({"type": "image", "image": b64_uri})

            content.append({"type": "text", "text": user_prompt})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]

            backend = model_dict.get("backend", "transformers")
            
            if backend == "gguf":
                result = self.run_inference_gguf(model_dict, messages, max_new_tokens, generation_config)
            else:
                result = self.run_inference_transformers(model_dict, messages, max_new_tokens, generation_config)

            parts = []
            if initial_tags and initial_tags.strip():
                parts.append(initial_tags.strip())
            parts.append(result)
            if end_tags and end_tags.strip():
                parts.append(end_tags.strip())
            
            final_output = " ".join(parts)
            return (final_output, user_prompt, system_prompt)

        finally:
            clean_vram(model_dict, unload=unload_when_done)


NODE_CLASS_MAPPINGS = {
    "Qwen3VL_ModelLoader": Qwen3VL_ModelLoader,
    "Qwen3VL_GenerationConfig": Qwen3VL_GenerationConfig,
    "Qwen3VL_Run": Qwen3VL_Run,
    "Qwen3VL_Run_Simple": Qwen3VL_Run_Simple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_ModelLoader": "RM-Qwen3-VL Loader (GGUF + HF)",
    "Qwen3VL_GenerationConfig": "RM-Qwen3-VL Config",
    "Qwen3VL_Run": "RM-Qwen3-VL Run",
    "Qwen3VL_Run_Simple": "RM-Qwen3-VL Run Simple",
}