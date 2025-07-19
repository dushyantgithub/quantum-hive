"""
Configuration settings for Quantum Hive
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BASE_DIR / "models"

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "int16"
}

# Speech-to-Text settings
STT_SETTINGS = {
    "engine": "vosk",  # Options: "vosk", "whisper"
    "model_path": str(MODELS_DIR / "vosk-model-small-en-us-0.15"),
    "language": "en-us",
    "timeout": 5.0
}

# Text-to-Speech settings
TTS_SETTINGS = {
    "engine": "espeak",  # Options: "espeak", "coqui", "pyttsx3"
    "voice": "en-us",
    "rate": 150,
    "volume": 0.8
}

# AI settings
AI_SETTINGS = {
    "primary": "local",  # Options: "local", "openai"
    "fallback": "openai",
    "temperature": 0.7,
    "max_tokens": 150
}

# OpenAI settings (if using API)
OPENAI_SETTINGS = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": "gpt-3.5-turbo",
    "max_tokens": 150
}

# Local LLM settings
LOCAL_LLM_SETTINGS = {
    "model_path": str(MODELS_DIR / "llama-2-7b-chat.gguf"),
    "context_length": 2048,
    "threads": 4
}

# UI settings
UI_SETTINGS = {
    "host": "localhost",
    "port": 3000,
    "debug": True
}

# Logging settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(BASE_DIR / "logs" / "quantum_hive.log")
}

# Create necessary directories
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        MODELS_DIR,
        BASE_DIR / "logs",
        BASE_DIR / "temp"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories() 