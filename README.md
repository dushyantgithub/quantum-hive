# 🧠 Quantum Hive

**Quantum Hive** is an offline-capable, Jarvis-like AI assistant built for Raspberry Pi 4 (8GB) and desktop. It uses local speech-to-text (Whisper), TinyLlama for AI, and Coqui TTS for voice responses.

---

## 🚀 Features

- 🎤 Offline speech-to-text using Whisper
- 💬 AI response from TinyLlama (local, chat-tuned)
- 🗣️ Voice responses using Coqui TTS (optimized for Pi)
- 👁️ Face-tracking avatar UI (Electron, in progress)
- 🌐 Raspberry Pi acts as an edge server, remotely accessible

---

## 📁 Project Structure

```
quantum-hive/
├── backend/
│   ├── main.py                # Core app logic (STT → AI → TTS loop)
│   ├── stt/                   # Speech-to-text (Whisper)
│   ├── tts/                   # Text-to-speech (Coqui)
│   ├── ai/                    # AI logic (TinyLlama)
│   └── utils/                 # Configs, helper tools
├── ui/                        # Placeholder for Electron app
├── deploy/                    # Scripts for Pi setup
├── tests/                     # Unit tests
├── README.md                  # Documentation
├── requirements.txt           # Python dependencies
└── start_dev.py               # Dev startup script
```

---

## ⚡️ Quick Start (Desktop or Pi)

### 1. **Clone and Enter Project**

```sh
git clone https://github.com/your-username/quantum-hive.git
cd quantum-hive
```

### 2. **Python Version**

- **You must use Python 3.10 or 3.11** (not 3.12+)
- On macOS: `brew install python@3.11`
- On Pi: `sudo apt-get install python3.11 python3.11-venv python3.11-dev`

### 3. **Create and Activate Virtual Environment**

```sh
python3.11 -m venv venv311
source venv311/bin/activate
```

### 4. **Install Dependencies**

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. **Install System Audio Tools**

- **macOS:** `brew install ffmpeg`
- **Raspberry Pi:** `sudo apt-get install ffmpeg aplay`

### 6. **Run the Assistant**

```sh
source venv311/bin/activate
python backend/main.py
```

---

## 🛠️ Troubleshooting

### **TTS/Audio Not Working?**

- Make sure you have a working speaker and your system is not muted.
- Try playing a test file:
  - macOS: `afplay /System/Library/Sounds/Glass.aiff`
  - Linux: `aplay /usr/share/sounds/alsa/Front_Center.wav`
- Make sure you are in Python 3.10/3.11 and have run `pip install TTS`.
- If you see errors about `pygame`, install it: `pip install pygame`.
- If you see `[TTS] Audio file is empty` or `[AUDIO] File does not exist`, check for TTS errors in the logs.
- You can force system audio playback by commenting out the `pygame` block in `tts_engine.py`.

### **STT Not Working?**

- Make sure your microphone is connected and not muted.
- Install Whisper with `pip install openai-whisper`.

### **Dependency Issues?**

- Always use the correct Python version (3.10 or 3.11).
- If you see `No matching distribution found for TTS`, you are likely on Python 3.12+ (not supported).
- If you see `No module named 'whisper'`, run `pip install openai-whisper` in your venv.
- If you see `No module named 'TTS'`, run `pip install TTS` in your venv.

---

## 🧪 Testing Individual Components

```sh
# Test STT
python backend/stt/whisper_engine.py
# Test TTS
python backend/tts/tts_engine.py
# Test AI
python backend/ai/gemma_text_engine.py
```

---

## 🐍 requirements.txt (Key Points)

- `TTS` (for Coqui TTS, Python 3.10/3.11 only)
- `openai-whisper` (for Whisper STT)
- `pygame` (for audio playback)
- `torch`, `transformers`, etc. (for LLMs)

---

## 🥧 Raspberry Pi Tips

- Use a lightweight TTS model: `tts_models/en/ljspeech/tacotron2-DDC`
- Lower sample rate (e.g., 16000 Hz) for faster synthesis
- Use a USB microphone and external speaker for best results
- Run `alsamixer` to check/adjust audio settings

---

## 📦 Future Add-ons

- Avatar with gesture animation & lip-sync
- Web dashboard for remote access

---

## 🛠 Contributors

- 🧠 You (the human behind the AI)
- 🤖 ChatGPT (your assistant for building it)

## 📜 License

MIT License

---

## 📝 Changelog

- 2024-07: Now uses only Whisper (STT), TinyLlama (AI), and Coqui TTS (TTS)
- 2024-07: Added debug logging for TTS/audio troubleshooting
- 2024-07: Updated requirements for Python 3.10/3.11 compatibility
- 2024-07: Improved README for setup and troubleshooting

## 🔑 Environment Variables & Credentials

- Store all credentials, API keys, and tokens in a `.env` file at the project root.
- Example `.env`:
  ```
  OPENAI_API_KEY=sk-...
  HUGGINGFACE_TOKEN=hf_...
  OTHER_SECRET=...
  ```
- The `.env` file is already included in `.gitignore` and will not be committed to GitHub.
- The app loads environment variables automatically using `os.getenv` (see `backend/utils/config.py`).

```

```

