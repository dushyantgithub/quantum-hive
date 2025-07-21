"""
Configuration settings for Quantum Hive
"""
import os
from pathlib import Path
import dotenv

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BASE_DIR / "models"

# Load .env file at project root
dotenv.load_dotenv(dotenv.find_dotenv())

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "int16"
}

# Speech-to-Text settings
STT_SETTINGS = {
    "engine": "whisper",  # Only Whisper is supported
    "whisper_model": "base",  # Options: "tiny", "base", "small", "medium", "large"
    "language": "en-us",
    "timeout": 5.0,
    "silence_threshold": 500,
    "silence_duration": 3.0,
    "min_audio_duration": 3.0
}

# Text-to-Speech settings
TTS_SETTINGS = {
    "engine": "coqui",  # Only Coqui TTS is supported
    "voice": "tts_models/en/ljspeech/tacotron2-DDC",  # Lightweight English model
    "rate": 16000,  # Lower sample rate for Pi
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
    "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "150")),
}

# Hugging Face settings (if using Hugging Face Hub)
HUGGINGFACE_SETTINGS = {
    "token": os.getenv("HUGGINGFACE_TOKEN", ""),
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

# Conversation memory settings
MEMORY_SETTINGS = {
    "history_file": str(BASE_DIR / "backend" / "utils" / "conversation_history.json"),
    "min_days": 1,   # Minimum age to keep (in days)
    "max_days": 7    # Maximum age to keep (in days)
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