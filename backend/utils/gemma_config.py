"""
TinyLlama Configuration (replaces Gemma 3 for speed)
Optimized settings for different hardware scenarios
"""

# Hardware-specific configurations
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TINYLLAMA_CONFIGS = {
    # For development/testing on Mac/PC
    "development": {
        "model_name": TINYLLAMA_MODEL,
        "use_cpu_only": True,
        "generation_config": {
            "max_new_tokens": 40,
            "temperature": 0.5,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "use_cache": True,
        },
        "torch_settings": {
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
            "device_map": "cpu",
        }
    },
    # For Raspberry Pi 4 deployment
    "raspberry_pi": {
        "model_name": TINYLLAMA_MODEL,
        "use_cpu_only": True,
        "generation_config": {
            "max_new_tokens": 25,
            "temperature": 0.3,
            "do_sample": True,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
        },
        "torch_settings": {
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
            "device_map": "cpu",
        }
    },
    # For GPU systems
    "gpu": {
        "model_name": TINYLLAMA_MODEL,
        "use_cpu_only": False,
        "generation_config": {
            "max_new_tokens": 80,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
        },
        "torch_settings": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        }
    },
    # Lightweight option (same as dev)
    "lightweight": {
        "model_name": TINYLLAMA_MODEL,
        "use_cpu_only": True,
        "generation_config": {
            "max_new_tokens": 30,
            "temperature": 0.5,
            "do_sample": True,
            "top_p": 0.8,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
        },
        "torch_settings": {
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
            "device_map": "cpu",
        }
    }
}

DEFAULT_CONFIG = "development"

def get_tinyllama_config(config_name=None):
    if config_name is None:
        config_name = DEFAULT_CONFIG
    if config_name not in TINYLLAMA_CONFIGS:
        print(f"Warning: Config '{config_name}' not found, using '{DEFAULT_CONFIG}'")
        config_name = DEFAULT_CONFIG
    return TINYLLAMA_CONFIGS[config_name]

def list_available_configs():
    return list(TINYLLAMA_CONFIGS.keys())

def detect_best_config():
    import torch
    import platform
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    has_gpu = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    is_pi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')
    if is_pi:
        return "raspberry_pi"
    elif has_gpu or has_mps:
        return "gpu"
    elif memory_gb < 6:
        return "lightweight"
    else:
        return "development"

if __name__ == "__main__":
    print("🔧 TinyLlama Configuration Options:")
    print("=" * 50)
    for config_name in list_available_configs():
        config = get_tinyllama_config(config_name)
        print(f"\n📋 {config_name.upper()}:")
        print(f"   Model: {config['model_name']}")
        print(f"   CPU Only: {config['use_cpu_only']}")
        print(f"   Max Tokens: {config['generation_config']['max_new_tokens']}")
    print(f"\n🎯 Auto-detected best config: {detect_best_config()}")
    import torch
    import platform
    import psutil
    print(f"\n💻 System Info:")
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}") 