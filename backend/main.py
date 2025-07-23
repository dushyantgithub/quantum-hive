"""
Quantum Hive - Main Application
Orchestrates Wake Word → STT → AI → TTS workflow

Requirements for wake word:
    pip install pvporcupine pyaudio
"""
import logging
import time
import signal
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pvporcupine
import pyaudio
import struct
import dotenv
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.tts.tts_engine import TTSEngine
from backend.ai.gemma_text_engine import TinyLlamaTextAIEngine
from backend.utils.config import LOGGING_SETTINGS, STT_SETTINGS, TTS_SETTINGS, MEMORY_SETTINGS
from backend.smart_home import GoogleHomeController

logging.basicConfig(
    level=getattr(logging, LOGGING_SETTINGS["level"]),
    format=LOGGING_SETTINGS["format"],
    handlers=[
        logging.FileHandler(LOGGING_SETTINGS["file"]),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class QuantumHive:
    """Main Quantum Hive application"""
    def __init__(self):
        self.running = False
        self.stt_engine = None
        self.tts_engine = None
        self.ai_engine = None
        self.google_home = None
        self.system_prompt = (
            "You are Quantum Hive, a helpful AI assistant running on a Raspberry Pi. "
            "You are designed to be conversational, helpful, and concise in your responses. "
            "You can help with various tasks and answer questions. Keep responses under 100 words unless asked for more detail."
        )
        self._initialize_components()
        self._setup_signal_handlers()

    def _initialize_components(self):
        try:
            logger.info("Initializing Quantum Hive components...")
            logger.info("Initializing Whisper STT engine...")
            self.stt_engine = WhisperSTTEngine(model_size=STT_SETTINGS["whisper_model"])
            logger.info("Initializing Coqui TTS engine...")
            self.tts_engine = TTSEngine(engine_type="coqui")
            logger.info("Initializing Google Home controller...")
            self.google_home = GoogleHomeController()
            # TinyLlama will be lazy-loaded when Advance mode is first used
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.conversation_history = self._load_recent_history()
            # If any messages are missing embeddings, add them
            for msg in self.conversation_history:
                if "user_embedding" not in msg:
                    msg["user_embedding"] = self._embed_text(msg["user"])
                if "ai_embedding" not in msg:
                    msg["ai_embedding"] = self._embed_text(msg["ai"])
            self._save_history()
            logger.info("All components initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _load_recent_history(self):
        """Load conversation history from the last 1-2 days"""
        history_file = MEMORY_SETTINGS["history_file"]
        min_days = MEMORY_SETTINGS["min_days"]
        max_days = MEMORY_SETTINGS["max_days"]
        now = datetime.utcnow()
        min_time = now - timedelta(days=max_days)
        max_time = now - timedelta(days=min_days)
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    all_msgs = json.load(f)
                except Exception:
                    all_msgs = []
            for msg in all_msgs:
                try:
                    ts = datetime.fromisoformat(msg["timestamp"])
                    if min_time <= ts <= now:
                        history.append(msg)
                except Exception:
                    continue
        return history

    def _save_history(self):
        """Save conversation history, pruning messages older than max_days"""
        history_file = MEMORY_SETTINGS["history_file"]
        max_days = MEMORY_SETTINGS["max_days"]
        now = datetime.utcnow()
        min_time = now - timedelta(days=max_days)
        pruned = [msg for msg in self.conversation_history if datetime.fromisoformat(msg["timestamp"]) >= min_time]
        # Ensure embeddings are saved
        for msg in pruned:
            if "user_embedding" not in msg:
                msg["user_embedding"] = self._embed_text(msg["user"])
            if "ai_embedding" not in msg:
                msg["ai_embedding"] = self._embed_text(msg["ai"])
        with open(history_file, "w") as f:
            json.dump(pruned, f, indent=2)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def _is_exit_command(self, text):
        exit_commands = [
            "exit", "quit", "stop", "goodbye", "bye", "shutdown", 
            "turn off", "power off", "end session"
        ]
        return any(cmd in text.lower() for cmd in exit_commands)

    def stop(self):
        """Stop the Quantum Hive application"""
        logger.info("Stopping Quantum Hive...")
        self.running = False
        # Clean shutdown of components if needed
        if hasattr(self, 'stt_engine') and self.stt_engine:
            # Stop any ongoing audio recording
            try:
                if hasattr(self.stt_engine, 'stop'):
                    self.stt_engine.stop()
            except Exception as e:
                logger.error(f"Error stopping STT engine: {e}")
        logger.info("Quantum Hive stopped successfully")

    def _is_memory_search_command(self, text):
        triggers = [
            "what did i say", "what was it", "remind me", "what did we talk about", "what did i mention"
        ]
        return any(trigger in text.lower() for trigger in triggers)

    def _get_mode_selection_prompt(self):
        """Get a random mode selection prompt"""
        mode_prompts = [
            "Which mode would you prefer, Master?",
            "Please select a mode, Master.",
            "What mode shall I activate for you, Master?",
            "Kindly tell me your desired mode, Master."
        ]
        return random.choice(mode_prompts)

    def _validate_mode_selection(self, user_input):
        """Validate and return the selected mode"""
        if not user_input:
            return None
        
        user_input_lower = user_input.lower().strip()
        
        # Check for Home mode
        if any(word in user_input_lower for word in ["home", "smart home", "house", "home mode"]):
            return "home"
        
        # Check for Advance mode  
        if any(word in user_input_lower for word in ["advance", "advanced", "advance mode", "chat", "conversation"]):
            return "advance"
        
        # Invalid selection
        return None

    def _handle_mode_selection(self):
        """Handle the mode selection process"""
        max_attempts = 3
        for attempt in range(max_attempts):
            # Ask for mode selection
            mode_prompt = self._get_mode_selection_prompt()
            print(f"🤖 **Quantum Hive:** {mode_prompt}")
            logger.info(f"Asking for mode selection: {mode_prompt}")
            self.tts_engine.speak(mode_prompt)
            
            # Listen for mode selection
            print("\n🎤 Listening for mode selection...")
            mode_input = self.stt_engine.listen_for_speech(timeout=15.0, min_record_time=1.0)
            
            if mode_input:
                print(f"🎤 **You said:** {mode_input}")
                logger.info(f"Mode selection input: {mode_input}")
                
                selected_mode = self._validate_mode_selection(mode_input)
                
                if selected_mode:
                    if selected_mode == "home":
                        confirmation = "Home mode activated, Master. Ready to control your smart home devices."
                    else:  # advance
                        confirmation = "Advance mode activated, Master. Ready for intelligent conversation."
                    
                    print(f"🤖 **Mode Selected:** {selected_mode.title()}")
                    logger.info(f"Mode selected: {selected_mode}")
                    self.tts_engine.speak(confirmation)
                    return selected_mode
                else:
                    # Invalid selection
                    error_msg = "Please choose either Home mode for smart home control, or Advance mode for AI conversation, Master."
                    print(f"🤖 **Invalid Selection:** {error_msg}")
                    self.tts_engine.speak(error_msg)
            else:
                # No speech detected
                timeout_msg = "I didn't hear your selection, Master. Please try again."
                print(f"🤖 **No Input:** {timeout_msg}")
                self.tts_engine.speak(timeout_msg)
        
        # After max attempts, default to advance mode
        default_msg = "Defaulting to Advance mode, Master."
        print(f"🤖 **Default Mode:** {default_msg}")
        logger.info("Defaulting to advance mode after max attempts")
        self.tts_engine.speak(default_msg)
        return "advance"

    def _ensure_ai_engine_loaded(self):
        """Lazy load TinyLlama AI engine when needed"""
        if self.ai_engine is None:
            print("🧠 **Loading AI Engine** - Please wait while I initialize the advanced conversation system...")
            logger.info("Lazy loading TinyLlama AI engine...")
            self.tts_engine.speak("Loading advanced AI system, Master. This will take a moment.")
            
            try:
                self.ai_engine = TinyLlamaTextAIEngine()
                print("✅ **AI Engine Ready** - Advanced conversation mode is now available")
                logger.info("TinyLlama AI engine loaded successfully")
                self.tts_engine.speak("Advanced AI system loaded successfully, Master. I'm ready for intelligent conversation.")
            except Exception as e:
                logger.error(f"Failed to load TinyLlama AI engine: {e}")
                self.tts_engine.speak("Sorry Master, I encountered an error loading the advanced AI system. Please try again.")
                raise

    def _handle_home_mode(self):
        """Handle Home mode - Smart home device control"""
        print("\n🏠 **Home Mode Active** - Ready for smart home commands")
        logger.info("Entering Home mode")
        
        # Listen for home control commands
        print("🎤 What would you like me to control in your home?")
        home_command = self.stt_engine.listen_for_speech(timeout=20.0, min_record_time=1.0)
        
        if home_command:
            print(f"🎤 **Home Command:** {home_command}")
            logger.info(f"Home command: {home_command}")
            
            # Process home command (placeholder for now)
            response = self._process_home_command(home_command)
            print(f"🏠 **Home Response:** {response}")
            logger.info(f"Home response: {response}")
            self.tts_engine.speak(response)
        else:
            no_command_msg = "I didn't hear a command, Master. Returning to standby mode."
            print(f"🏠 **No Command:** {no_command_msg}")
            self.tts_engine.speak(no_command_msg)

    def _process_home_command(self, command):
        """Process smart home commands using Google Home API"""
        command_lower = command.lower()
        
        try:
            # Light controls
            if any(word in command_lower for word in ["light", "lights"]):
                # Determine location if specified
                location = "all"  # default
                for room in ["living room", "bedroom", "kitchen"]:
                    if room in command_lower:
                        location = room.replace(" ", "_")
                        break
                
                # Determine action
                if any(word in command_lower for word in ["on", "turn on", "switch on"]):
                    return self.google_home.control_lights("on", location)
                elif any(word in command_lower for word in ["off", "turn off", "switch off"]):
                    return self.google_home.control_lights("off", location)
                else:
                    return "Light command received, Master. Please specify whether to turn them on or off."
                    
            # Temperature controls
            elif any(word in command_lower for word in ["temperature", "thermostat", "heat", "cool"]):
                # Extract specific temperature if mentioned
                import re
                temp_match = re.search(r'(\d+)\s*degrees?', command_lower)
                target_temp = int(temp_match.group(1)) if temp_match else None
                
                if target_temp:
                    return self.google_home.control_temperature("set", target_temp)
                elif any(word in command_lower for word in ["increase", "up", "warmer", "heat"]):
                    return self.google_home.control_temperature("increase")
                elif any(word in command_lower for word in ["decrease", "down", "cooler", "cool"]):
                    return self.google_home.control_temperature("decrease")
                else:
                    return "Temperature command received, Master. Please specify to increase, decrease, or set to a specific temperature."
                    
            # Fan controls
            elif any(word in command_lower for word in ["fan", "ceiling fan"]):
                # Determine speed if specified
                speed = None
                if "low" in command_lower:
                    speed = "low"
                elif "medium" in command_lower:
                    speed = "medium"
                elif "high" in command_lower:
                    speed = "high"
                
                if any(word in command_lower for word in ["on", "start"]):
                    return self.google_home.control_fan("on", speed)
                elif any(word in command_lower for word in ["off", "stop"]):
                    return self.google_home.control_fan("off")
                else:
                    return "Fan command received, Master. Please specify whether to turn it on or off."
            
            # Unknown command
            else:
                return ("I understand you want to control something in your home, Master. "
                       "I can control lights, temperature, and fans. Please specify what you'd like me to control.")
                       
        except Exception as e:
            logger.error(f"Error processing home command: {e}")
            return f"Sorry Master, I encountered an error processing your home command: {str(e)}"

    def _handle_advance_mode(self, last_user_input):
        """Handle Advance mode - AI conversation with TinyLlama"""
        print("\n🧠 **Advance Mode Active** - Ready for intelligent conversation")
        logger.info("Entering Advance mode")
        
        # Ensure AI engine is loaded (lazy loading)
        self._ensure_ai_engine_loaded()
        
        logger.info("Wake word detected. Listening for speech...")
        user_input = self.stt_engine.listen_for_speech(timeout=30.0, min_record_time=3.0)
        
        if user_input:
            print(f"\n🎤 **You said:** {user_input}")
            logger.info(f"User said: {user_input}")
            
            if self._is_exit_command(user_input):
                print(f"\n🛑 **Exit command detected:** {user_input}")
                logger.info("Exit command detected")
                self.stop()
                return
                
            if self._is_memory_search_command(user_input) and last_user_input:
                logger.info("Memory search command detected.")
                try:
                    results = self.search_memory(last_user_input, top_n=1)
                    if results:
                        mem = results[0]
                        response = f"You said: {mem['user']}. I replied: {mem['ai']}"
                    else:
                        response = "I don't remember anything relevant from the last week."
                    print(f"🤖 **Memory Search Result:** {response}")
                    self.tts_engine.speak(response)
                    return last_user_input
                except Exception as e:
                    logger.error(f"Memory search failed: {e}")
                    response = "Sorry Master, I had trouble searching my memory."
                    self.tts_engine.speak(response)
                    return last_user_input
                
            logger.info("Generating AI response...")
            # Pass recent history to TinyLlama
            history_pairs = [(msg["user"], msg["ai"]) for msg in self.conversation_history[-6:]]
            ai_response = self.ai_engine.generate_response(
                user_input,
                system_prompt=self.system_prompt,
                history=history_pairs
            )
            print(f"🤖 **AI Response:** {ai_response}")
            logger.info(f"AI response: {ai_response}")
            logger.info("Speaking response...")
            self.tts_engine.speak(ai_response)
            
            # Store the exchange with timestamp
            self.conversation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "user": user_input,
                "ai": ai_response,
                "user_embedding": self._embed_text(user_input),
                "ai_embedding": self._embed_text(ai_response)
            })
            self._save_history()
            
            print(f"\n🎧 **Say the wake word to activate again...**")
            return user_input
        else:
            logger.debug("No speech detected")
            return last_user_input

    def _embed_text(self, text):
        """Embed a single text string using the SentenceTransformer."""
        return self.embedder.encode(text, convert_to_tensor=True).tolist()

    def _listen_for_wake_word(self):
        """Continuously listen for the custom 'Activate Hive' wake word using the .ppn file and Picovoice access key."""
        import scipy.signal
        from backend.utils.config import AUDIO_SETTINGS
        
        dotenv.load_dotenv(dotenv.find_dotenv())
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_ACCESS_KEY not set in environment or .env file.")
        keyword_path = str(Path(__file__).parent / "porcupine" / "Activate_hive.ppn")
        porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path], sensitivities=[0.7])
        
        # Get device sample rate and calculate resampling parameters
        device_sample_rate = AUDIO_SETTINGS["sample_rate"]  # 44100 Hz
        porcupine_sample_rate = porcupine.sample_rate  # 16000 Hz
        input_device = AUDIO_SETTINGS.get("input_device")
        
        # Calculate frame sizes for resampling
        device_frame_length = int(porcupine.frame_length * device_sample_rate / porcupine_sample_rate)
        
        pa = pyaudio.PyAudio()
        
        # Open audio stream with device's native sample rate
        open_kwargs = dict(
            rate=device_sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=device_frame_length
        )
        if input_device is not None:
            open_kwargs["input_device_index"] = input_device
            
        audio_stream = pa.open(**open_kwargs)
        
        print(f"\n👂 Listening for wake word: 'Activate Hive' (device: {device_sample_rate}Hz → porcupine: {porcupine_sample_rate}Hz)...")
        logger.info(f"Listening for wake word with resampling {device_sample_rate}Hz → {porcupine_sample_rate}Hz")
        detected = False
        try:
            while not detected and self.running:
                # Read audio at device sample rate
                pcm_data = audio_stream.read(device_frame_length, exception_on_overflow=False)
                pcm_array = struct.unpack_from("h" * device_frame_length, pcm_data)
                
                # Resample to Porcupine's required sample rate
                resampled = scipy.signal.resample(pcm_array, porcupine.frame_length)
                resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
                
                # Process with Porcupine
                result = porcupine.process(resampled_int16)
                if result >= 0:
                    print("\n🟢 Wake word 'Activate Hive' detected!")
                    logger.info("Wake word detected!")
                    detected = True
                    break
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
        finally:
            audio_stream.close()
            pa.terminate()
            porcupine.delete()
        return detected

    def start(self):
        logger.info("Starting Quantum Hive...")
        self.running = True
        # No initial greeting, just listen for wake word
        logger.info("Quantum Hive is ready!")
        try:
            last_user_input = None
            while self.running:
                # Wake word loop
                if not self._listen_for_wake_word():
                    continue
                
                # Handle mode selection after wake word
                selected_mode = self._handle_mode_selection()
                
                try:
                    if selected_mode == "home":
                        # Home mode - Smart home control
                        self._handle_home_mode()
                    else:
                        # Advance mode - AI conversation
                        result = self._handle_advance_mode(last_user_input)
                        if result:  # Update last_user_input if we got one
                            last_user_input = result
                            
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    try:
                        self.tts_engine.speak("I encountered an error. Please try again.")
                    except:
                        pass
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            self.stop()

    def search_memory(self, query, top_n=5):
        """Semantic search conversation history for a query string (last 7 days)"""
        query_emb = self._embed_text(query)
        scored = []
        for msg in self.conversation_history:
            # Compare to both user and ai text
            user_sim = util.cos_sim(np.array(query_emb), np.array(msg["user_embedding"]))
            ai_sim = util.cos_sim(np.array(query_emb), np.array(msg["ai_embedding"]))
            max_sim = max(float(user_sim), float(ai_sim))
            scored.append((max_sim, msg))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [msg for sim, msg in scored[:top_n] if sim > 0.4]  # Only return relevant matches

def main():
    print("🧠 Quantum Hive - AI Assistant")
    print("=" * 40)
    try:
        quantum_hive = QuantumHive()
        quantum_hive.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Failed to start Quantum Hive: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 