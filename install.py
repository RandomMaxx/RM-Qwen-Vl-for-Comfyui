"""
Installation Script for Qwen3-VL ComfyUI Node
Executes pip install within the current Python environment.
Includes smart detection for Hardware Acceleration (CUDA/Metal).
"""

import sys
import subprocess
import importlib.util
import os
import platform
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def is_installed(package):
    try:
        subprocess.check_output([sys.executable, '-m', 'pip', 'show', package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_llama_cpp_smartly():
    """
    Attempts to install llama-cpp-python with GPU acceleration (CUDA/Metal)
    if the hardware supports it.
    """
    package = "llama-cpp-python"
    
    print(f"{YELLOW}🔍 Checking hardware for GGUF acceleration...{RESET}")
    
    # 1. Check for NVIDIA GPU (CUDA)
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # 2. Prepare Environment
    env = os.environ.copy()
    install_cmd_base = [sys.executable, "-m", "pip", "install", package, "--no-cache-dir", "--force-reinstall", "--upgrade"]
    
    # 3. Determine Installation Strategy
    if has_cuda:
        print(f"{GREEN}🚀 NVIDIA GPU detected. Installing with CUDA support...{RESET}")
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        try:
            subprocess.check_call(install_cmd_base, env=env)
            return True
        except subprocess.CalledProcessError:
            print(f"{RED}⚠️ CUDA install failed. Falling back to CPU...{RESET}")

    elif platform.system() == "Darwin":
        print(f"{GREEN}🍎 macOS detected. Installing with Metal support...{RESET}")
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        try:
            subprocess.check_call(install_cmd_base, env=env)
            return True
        except subprocess.CalledProcessError:
             pass
    
    # 4. Fallback or if already installed/no GPU
    if not is_installed(package):
        print(f"{YELLOW}⚙️ Standard CPU installation for {package}...{RESET}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{GREEN}✅ {package} is already installed.{RESET}")

def check_and_install():
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    print(f"{YELLOW}🚀 Starting Qwen3-VL Dependency Check...{RESET}")
    
    if not requirements_path.exists():
        print(f"{RED}❌ requirements.txt not found!{RESET}")
        return

    # 1. Install Standard Dependencies via pip
    try:
        print(f"{YELLOW}📦 Installing base dependencies...{RESET}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
        )
        print(f"{GREEN}✅ Base dependencies installed.{RESET}")
    except subprocess.CalledProcessError as e:
        print(f"{RED}❌ Installation failed: {e}{RESET}")
        return

    # 2. Smart Install for GGUF (The tricky part)
    try:
        install_llama_cpp_smartly()
    except Exception as e:
        print(f"{RED}❌ GGUF Setup failed: {e}{RESET}")

    # 3. Verification Step
    print(f"{YELLOW}🔍 Verifying critical modules...{RESET}")
    
    critical_libs = ["qwen_vl_utils", "bitsandbytes", "transformers", "llama_cpp"]
    missing = []
    
    for lib in critical_libs:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
    
    if missing:
        print(f"{RED}❌ The following modules failed to install properly: {', '.join(missing)}{RESET}")
        print(f"{YELLOW}👉 Please try running: pip install -r requirements.txt manually.{RESET}")
    else:
        print(f"{GREEN}✨ All Systems Go! Qwen3-VL Node is ready to run.{RESET}")

if __name__ == "__main__":
    check_and_install()