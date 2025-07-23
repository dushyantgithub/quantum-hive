# 🧠 Quantum Hive

**Quantum Hive** is an offline-capable, Jarvis-like AI assistant built for Raspberry Pi 4 (8GB) and desktop environments. It uses local speech-to-text (Whisper), TinyLlama for AI chat responses, and Coqui TTS for voice output.

---

## 🚀 Features

- 🎤 Offline speech-to-text using Whisper
- 💬 Local AI responses using TinyLlama (chat-tuned)
- 🗣️ Voice responses using Coqui TTS (optimized for Pi)
- 🛡️ Offline wake word detection (privacy-first)
- 🗨️ Random activation phrases after wake word
- 👁️ Face-tracking avatar UI (Electron, in progress)
- 🌐 Raspberry Pi acts as an edge server, remotely accessible
- 🛠️ Custom wake word support (via Porcupine)

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

### 1. Clone and Enter Project

```bash
git clone https://github.com/your-username/quantum-hive.git
cd quantum-hive
```

### 2. Python Version

> ⚠️ You must use **Python 3.10 or 3.11**  
> Python 3.12+ is not supported by Coqui TTS.

- macOS:
  ```bash
  brew install python@3.11
  ```
- Raspberry Pi:
  ```bash
  sudo apt-get install python3.11 python3.11-venv python3.11-dev
  ```

### 3. Create and Activate Virtual Environment

```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install System Audio Tools

- macOS:
  ```bash
  brew install ffmpeg
  ```
- Raspberry Pi:
  ```bash
  sudo apt-get install ffmpeg aplay
  ```

**Audio Device Configuration (Raspberry Pi):**
- **Input (Microphone):** USB PnP Sound Device (usually auto-detected)
- **Output (Speaker):** Built-in headphone jack (device: `hw:2,0`)
- The app is pre-configured to use these devices in `backend/utils/config.py`:
  ```python
  AUDIO_SETTINGS = {
      # ...
      "input_device": 1,         # USB mic index
      "output_device": "hw:2,0" # Headphone jack for playback
  }
  ```

### 6. Run the Assistant

```bash
source venv311/bin/activate
python backend/main.py
```

---

## 🎙️ Wake Word Setup (Porcupine)

### 1. Get a Free Access Key

- Sign up at: https://console.picovoice.ai/
- Copy your **Access Key**
- Add it to your `.env` file:
  ```env
  PICOVOICE_ACCESS_KEY=YOUR_KEY_HERE
  ```

### 2. Add Wake Word File (`.ppn`)

- Generate or download a `.ppn` file for your keyword ("Activate Hive" or custom)
- Target platform: Raspberry Pi (ARM32/ARM64) or macOS
- Place it here:
  ```
  backend/porcupine/Activate_hive.ppn
  ```

### 3. Install Dependencies

```bash
pip install pvporcupine pyaudio
```

### 4. How It Works

- Assistant runs in silent listening mode.
- When it hears **"Activate Hive"**, it responds with a **random activation phrase**, like:
  - "Activating Hive Mind"
  - "At your service, master"
  - "Activating Quantum"
  - "Booting Hive Mind"
- Then it listens for your input, processes it, responds, and returns to passive wake mode.

---

## 🛠️ Troubleshooting

### 🔊 TTS / Audio Not Working?

- Ensure speakers are connected to the **headphone jack** (not HDMI) and unmuted.
- Test with:
  - macOS:
    ```bash
    afplay /System/Library/Sounds/Glass.aiff
    ```
  - Linux (headphone jack):
    ```bash
    aplay -D hw:2,0 /usr/share/sounds/alsa/Front_Center.wav
    ```
- If `pygame` throws errors like `ALSA: Couldn't open audio device: Unknown error 524`, **this is normal** on headless or Pi setups. The app will automatically fall back to `aplay` with the correct device.
- If Coqui TTS fails or returns empty audio:
  - Check for model download errors
  - Ensure the output device is set to `hw:2,0` in `AUDIO_SETTINGS`.

### 🎤 STT Not Working?

- Check mic is connected and system input is correct
- Install Whisper if missing:
  ```bash
  pip install openai-whisper
  ```

### 🧩 Dependency Issues?

- Make sure you're using **Python 3.10 or 3.11**
- Common fixes:
  ```bash
  pip install openai-whisper
  pip install TTS
  pip install pygame
  ```

---

## 🧪 Test Individual Components

```bash
# Test STT
python backend/stt/whisper_engine.py

# Test TTS
python backend/tts/tts_engine.py

# Test AI
python backend/ai/gemma_text_engine.py
```

---

## 🐍 Key Dependencies (requirements.txt)

- `TTS` (Coqui TTS)
- `openai-whisper` (Whisper STT)
- `torch`, `transformers` (for TinyLlama)
- `pygame` (Audio playback)
- `pvporcupine`, `pyaudio` (Wake word detection)

---

## 🥧 Raspberry Pi Optimization Tips

- Use a light TTS model:
  ```
  tts_models/en/ljspeech/tacotron2-DDC
  ```
- Set audio sample rate to 16000 Hz for better performance
- Use **USB mic** and **external speaker**
- Check volume/input using:
  ```bash
  alsamixer
  ```

---

## 📦 Planned Features

- 🎭 Animated Avatar with facial gestures
- 🌐 Web dashboard for remote interaction
- 💡 Smart home integration (Google Home API)

---

## 👥 Contributors

- 🧠 Dushyant Singh (the human behind the AI)
- 🤖 ChatGPT + Cursor (Vibe code + reasoning + research)

---

## 📜 License

[MIT License](LICENSE)

---

## 📝 Changelog

- **2024-07**: Migrated to Whisper + TinyLlama + Coqui TTS
- **2024-07**: Added wake word detection using Porcupine
- **2024-07**: Added debug logging for audio issues
- **2024-07**: Cleaned up README and refactored structure

---

## 🔐 Environment Variables

All secrets and tokens should go in a `.env` file at the root:

```
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
PICOVOICE_ACCESS_KEY=...
```

The app automatically loads this using `os.getenv()` (see `backend/utils/config.py`).  
`.env` is ignored by Git via `.gitignore`.

```

---
```

