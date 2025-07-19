# 🧠 Quantum Hive

**Quantum Hive** is an offline-capable, Jarvis-like AI assistant built on a Raspberry Pi 4 (8GB). It uses local speech-to-text and text-to-speech systems, processes natural language using either local or API-based LLMs, and responds with a speaking avatar UI.

---

## 🚀 Features

- 🎤 Offline speech-to-text using Vosk
- 💬 AI response from either OpenAI (cloud) or quantized LLM (local)
- 🗣️ Voice responses using eSpeak NG or Coqui TTS
- 👁️ Face-tracking avatar UI built with Electron + Three.js (in progress)
- 🌐 Raspberry Pi acts as an edge server, remotely accessible
- 🔌 Google Home integration planned for device control

---

## 📁 Project Structure Created

Here's what we've set up for you:

```
quantum-hive/
├── backend/
│   ├── main.py                # ✅ Core app logic (STT → AI → TTS loop)
│   ├── stt/                   # ✅ Speech-to-text (Vosk)
│   │   ├── __init__.py
│   │   └── vosk_engine.py
│   ├── tts/                   # ✅ Text-to-speech (eSpeak/Coqui/pyttsx3)
│   │   ├── __init__.py
│   │   └── tts_engine.py
│   ├── ai/                    # ✅ AI logic (local LLM + fallback)
│   │   ├── __init__.py
│   │   └── local_llm.py
│   └── utils/                 # ✅ Configs, helper tools
│       ├── __init__.py
│       └── config.py
├── ui/                        # 📁 Placeholder for future Electron app
│   └── electron-app/
├── deploy/                    # ✅ Scripts for Pi setup
│   └── setup_pi.sh
├── tests/                     # ✅ Unit tests
│   └── test_stt.py
├── README.md                  # ✅ Updated documentation
├── requirements.txt           # ✅ Python dependencies
├── .gitignore                # ✅ Git ignore rules
└── start_dev.py              # ✅ Development startup script
```

## 🚀 Key Features Implemented

### **Speech-to-Text (STT)**

- ✅ Vosk engine with offline speech recognition
- ✅ Microphone input handling
- ✅ Silence detection
- ✅ Audio file transcription

### **Text-to-Speech (TTS)**

- ✅ Multiple engine support (eSpeak, Coqui, pyttsx3)
- ✅ Audio file generation and playback
- ✅ Cross-platform audio support

### **AI Processing**

- ✅ Local LLM with llama.cpp integration
- ✅ Simple fallback AI for when models aren't available
- ✅ Configurable system prompts
- ✅ Conversation history support

### **Configuration & Utilities**

- ✅ Centralized configuration management
- ✅ Logging setup
- ✅ Path management
- ✅ Error handling

### **Deployment**

- ✅ Raspberry Pi setup script
- ✅ Systemd service creation
- ✅ Audio configuration
- ✅ Performance optimizations

## 🧪 Next Steps

1. **Test the setup locally:**

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Test individual components
   python backend/stt/vosk_engine.py
   python backend/tts/tts_engine.py
   python backend/ai/local_llm.py

   # Run the full application
   python start_dev.py
   ```

2. **Download Vosk model:**

   ```bash
   mkdir models
   cd models
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip
   ```

3. **For Raspberry Pi deployment:**
   ```bash
   chmod +x deploy/setup_pi.sh
   ./deploy/setup_pi.sh
   ```

## 🔧 Quick Start

````bash
# Clone the repo
git clone https://github.com/your-username/quantum-hive.git
cd quantum-hive

# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the assistant
python backend/main.py
📦 Future Add-ons

Local LLM with llama.cpp or ggml
MQTT/HTTP device control via Google Home
Avatar with gesture animation & lip-sync
Web dashboard for remote access
🛠 Contributors

🧠 You (the human behind the AI)
🤖 ChatGPT (your assistant for building it)
📜 License

MIT License


---

### ✅ Next Steps

1. Do you want me to generate this folder with template `.py` files and a working `main.py` that ties STT → AI → TTS?
2. Would you like to start with **Vosk + eSpeak** for first testing?

Let me know if you'd like this as a downloadable ZIP or GitHub-ready format.

🎉 **AMAZING! Your Quantum Hive is working perfectly!**

## 🧪 Test Results Summary

✅ **All components are working:**

1. **Speech-to-Text (STT)**: ✅ Vosk engine successfully transcribed your speech
2. **Text-to-Speech (TTS)**: ✅ pyttsx3 engine spoke the responses clearly
3. **AI Processing**: ✅ Simple AI engine provided responses
4. **Full Integration**: ✅ Complete STT → AI → TTS workflow working

## 🤔 What Just Happened

The system successfully:
- **Listened** to your speech using Vosk STT
- **Processed** your input through the AI engine
- **Spoke back** responses using pyttsx3 TTS
- **Detected exit commands** and shut down gracefully

## 🔧 Current Configuration

- **STT**: Vosk with offline English model
- **TTS**: pyttsx3 (macOS native speech synthesis)
- **AI**: Simple rule-based engine (fallback mode)
- **Audio**: Real-time microphone input and speaker output

## 🚀 Next Steps for Enhancement

1. **Improve AI Responses**: Install llama-cpp-python for better AI
   ```bash
   pip install llama-cpp-python
````

2. **Add More Commands**: Extend the SimpleAIEngine with more responses

3. **Test on Raspberry Pi**: Use the deployment script we created

4. **Add Voice Commands**: Implement specific voice commands for device control

## 🎉 Congratulations!

Your Quantum Hive AI assistant is now fully functional! You can:

- Speak to it and get voice responses
- Have basic conversations
- Use voice commands to exit ("goodbye", "exit", etc.)

The system is ready for further development and can be deployed to your Raspberry Pi when you're ready! 🚀

```

```

