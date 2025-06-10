# quantum-hive

# 🧠 Jarvis AI Assistant on Raspberry Pi

A fully functional **Jarvis-like AI assistant** built using a **Raspberry Pi 4B (8GB)**, touch screen, mic, camera, and speaker. This assistant can **listen**, **think**, **respond**, and even **control smart home devices**—all while displaying an interactive 3D avatar interface with animations and face tracking.

## 💡 Project Overview

This open-source project aims to replicate a real-time, responsive AI assistant similar to J.A.R.V.I.S from Iron Man, using only low-cost, open tools and hardware.

### 🔧 Core Features

- 🎙️ **Speech-to-Text** (STT) using free, offline-capable models like [Vosk](https://alphacephei.com/vosk/)
- 🧠 **AI Brain** powered by Open Source LLMs or lightweight APIs
- 🔊 **Text-to-Speech** (TTS) using models like Coqui TTS or Pico TTS
- 💬 **Natural Conversations** with context awareness
- 👁️ **Face Tracking** with the Pi camera (OpenCV + face landmarks)
- 📱 **Smart Home Integration** via Google Home API / Matter API
- 🖥️ **Electron App UI** with a 3D animated avatar (Three.js / Anime.js)
- 🌐 **Remote Access** by turning Pi into a secure web-accessible server

---

## 🖥️ Hardware Requirements

- Raspberry Pi 4B (8GB RAM)
- 5.5" Touchscreen display
- USB Microphone
- USB Speaker or 3.5mm Audio Output
- Raspberry Pi Camera Module
- SD Card (32GB+ recommended)
- Power Supply
- Optional: External casing, cooling fan, and power bank

---

## 📦 Software Stack

| Component         | Technology                              |
|------------------|------------------------------------------|
| OS               | Raspberry Pi OS Lite (64-bit)            |
| STT              | Vosk + Python bindings                   |
| AI Engine        | OpenAI API or Local LLM (e.g. GPT4All)   |
| TTS              | Coqui TTS / PicoTTS                      |
| Face Tracking    | OpenCV + Dlib / Mediapipe                |
| UI Frontend      | Electron + Three.js / Anime.js           |
| Backend API      | Node.js or Python Flask server           |
| Home Control     | Google Home Graph API / Matter Protocol  |
| Remote Access    | Ngrok / Tailscale / Custom SSH Tunnel    |

---

## 🚀 Getting Started

### 🔌 1. Hardware Setup

1. Install Raspberry Pi OS on SD card.
2. Connect touchscreen, mic, speaker, and camera.
3. Enable camera and audio in `raspi-config`.

### 🧱 2. Install Dependencies

```bash
sudo apt update && sudo apt upgrade
sudo apt install python3-pip portaudio19-dev libasound2-dev libatlas-base-dev
pip3 install vosk openai flask coqui-tts opencv-python


(Install Electron and other UI packages in the ui/ directory separately)

⚙️ 3. Configure Environment
Create a .env file in your root project:

OPENAI_API_KEY=your-key
GOOGLE_HOME_API_TOKEN=your-google-token
USE_LOCAL_LLM=false
TTS_ENGINE=coqui
🧠 4. Run the Assistant
python3 main.py
Or to launch the UI version:

npm install
npm start
📡 Remote Access

Set up Tailscale or Ngrok for secure remote access:

Tailscale guide
Ngrok
🤖 Avatar + UI

Located in the ui/ folder. Built using:

Electron
Three.js (3D avatar)
Anime.js (gesture animations)
Socket connection to Pi backend
🏠 Smart Home Integration

Control lights, devices, and more via:

Google Home Graph API
(Future) Matter Protocol via ESP32 + Zigbee integration
📷 Face Tracking

Real-time face tracking using:

OpenCV for face detection
Dlib or Mediapipe for landmarks
Camera controls avatar gaze
🧰 Future Additions

Emotion detection via tone analysis
LLM-based long-term memory
Wake word detection ("Initiate Hive Mind")
Custom device control over MQTT or Zigbee
Whisper STT integration for better accuracy
🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.

📄 License

This project is licensed under the MIT License.

👤 Creator

Made with ❤️ by Dushyant Rathore

Inspired by Tony Stark, powered by open source.

---


