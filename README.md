# **Qwen3-VL ComfyUI Node Suite (High Performance)**

A unified, production-grade ComfyUI node suite for **Qwen3-VL** Vision-Language models.

This suite has been re-engineered for **reliability**, **speed**, and **ease of use**. It features a **Unified Loader** that automatically handles both standard Transformers (High Precision) and GGUF (Low VRAM/High Speed) models, along with powerful image management tools.

## **🚀 Key Features**

### **🛡️ Unified Backend Support**

* **Auto-Detect:** The loader automatically identifies if a model is GGUF or Transformers based on the file/folder selected.
* **Transformers:** Full support for 4-bit/8-bit bitsandbytes quantization and Flash Attention 2.
* **GGUF:** Native llama-cpp-python integration for running 8B+ models on consumer GPUs (6GB-8GB VRAM) or Apple Silicon.

### **📦 Dedicated Model Manager**

* **GUI-Based Management:** Download, Check Status, or Remove models directly from ComfyUI using the Model Manager node.
* **JSON Config:** All model URLs and settings are defined in qwen\_models.json for easy updates.

### **🖼️ Advanced Image Handling**

* **Directory Scanning:** The Image Loader can scan entire local folders, filter by name, and cycle through images by index.
* **History System:** Remembers your recently used folders for quick access.
* **Smart Rescaling:** Dedicated Image Rescaler ensures optimal resolution matching for VLM inputs.

### **🛠️ Production Ready**

* **Logging:** Detailed operation logs saved to logs/qwen3\_vl.log.
* **CUDA Safety:** Strict VRAM cleanup, garbage collection, and OOM protection.
* **Prompt Engineering:** Built-in "Thinking", "Describe", and "Analyze" presets.

## **📥 Installation**

### **Method 1: Manual (Recommended)**

1. Navigate to your ComfyUI custom nodes directory:
```
   cd ComfyUI/custom\_nodes/
```
3. Clone this repository:
```
   git clone \[https://github.com/yourusername/ComfyUI-Qwen3-VL.git](https://github.com/yourusername/ComfyUI-Qwen3-VL.git)  
   cd ComfyUI-Qwen3-VL
```
5. **Install Dependencies**:
```
   * **Windows (Portable):**  
     ..\\..\\..\\python\_embeded\\python.exe -m pip install -r requirements.txt
```
```
   * **Linux / Mac / Venv:**  
     pip install -r requirements.txt
```
\[!NOTE]  
For GGUF support on NVIDIA GPUs, ensure llama-cpp-python is compiled with CUDA support. If pre-built wheels fail, refer to the llama-cpp-python documentation.

## **📂 Folder Structure**

The nodes expect models in specific directories inside your ComfyUI models folder.  
Logs and Configs are stored inside the custom node folder.  
```
ComfyUI/  
├── models/  
│   └── VLM/                        <-- Main Model Directory  
│       ├── Qwen/  
│       │   └── Qwen3-VL-8B-Inst/   <-- Standard HF Models (Folders)  
│       ├── qwen3-vl-8b-q4\_k\_m.gguf <-- GGUF Models (Single Files)  
│       └── mmproj-model-f16.gguf   <-- GGUF Vision Projectors  
│  
└── custom\_nodes/  
└── ComfyUI-Qwen3-VL/  
├── qwen\_models.json        <-- Model Definitions (URLs/Names)  
├── qwen\_history.json       <-- User Folder History  
└── logs/  
└── qwen3\_vl.log        <-- Debug \& Error Logs
```
## **🧩 Node Overview**

### **1. Management \& Loading**

#### **RM-Qwen3-VL Model Manager**

* **Role:** Your control center for model files.
* **Inputs:**

  * model\_name: Select from a predefined list (starred items \* are not yet installed).
  * action: Download (fetches from HF), Check Status, or Remove (deletes local files).

* **Note:** Supports both GGUF (files) and Transformers (folders) downloading.

#### **Qwen3-VL Loader (GGUF + HF)**

* **Role:** The universal model loader.
* **Inputs:**

  * model: Select any installed model. The node auto-detects the backend.
  * quantization: For Transformers models (4bit/8bit/None).
  * gpu\_layers\_gguf: For GGUF models (-1 = Offload All to GPU).

### **2. Image Tools**

#### **RM-Qwen3-VL Image Loader**

* **Role:** A powerful alternative to the standard "Load Image" node.
* **Features:**

  * **Directory Scan:** Point it to a folder (directory) to load images sequentially.
  * **Quick Select:** Dropdown menu remembers your last 15 used paths.
  * **Filtering:** Use filter to find specific filenames (e.g., "landscape").
  * **Indexing:** Use index to cycle through images in a folder.

#### **RM-Qwen3-VL Image Rescaler**

* **Role:** Prepares images for the VLM.
* **Features:** Resizes images to standard resolutions (e.g., 1024x1024) while maintaining aspect ratio and ensuring dimensions are divisible by the model's patch size (default modulo 16/32).

### **3. Inference**

#### **RM-Qwen3-VL Run**

* **Role:** The main brain. Generates text from images.
* **Inputs:**

  * caption\_type: Choose presets like "Descriptive", "Thinking" (for COT models), "Danbooru Tags", or "JSON".
  * prompt\_mode: Prepend your custom prompt or Overwrite the system prompt.
  * system\_prompt: Fully customizable system instructions.
  * boolean flags: Toggles for specific tasks (e.g., "Read Text", "Analyze Lighting", "Describe Clothing").

## **⚠️ Common Issues \& Troubleshooting**

### **1. "RuntimeError: CUDA Out of memory"**

* **Transformers:** Use the Loader to set quantization to **4bit**.
* **GGUF:** Reduce n\_ctx (context size) or use a smaller quantization file (e.g., Q4\_K\_S).
* **General:** Enable unload\_when\_done in the Run node.

### **2. "GGUF Model missing mmproj file"**

* **Cause:** GGUF models require a separate "Vision Projector" file (usually named mmproj-....gguf).
* **Fix:** Use the **Model Manager** node to download the GGUF model; it automatically fetches the required mmproj file defined in qwen\_models.json.

### **3. "No models found"**

* **Fix:** Connect the **Model Manager** node, select a model, set action to **Download**, and run the queue once.

### **4. Logs \& Debugging**

* Check custom\_nodes/ComfyUI-Qwen3-VL/logs/qwen3\_vl.log for detailed error messages and download progress.

## **📜 License**

This custom node is open-source. The Qwen3-VL models are subject to the original license by the Qwen Team (Alibaba Cloud).

