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

# Load environment variables first
dotenv.load_dotenv(dotenv.find_dotenv())

# Debug: Print the loaded endpoint
print(f"🔍 [DEBUG] Loaded LOCAL_LLM_ENDPOINT: {os.getenv('LOCAL_LLM_ENDPOINT', 'NOT SET')}")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.tts.tts_engine import TTSEngine
# Removed TinyLlamaTextAIEngine - using API instead
from backend.ai.gemma_api_client import GemmaAPIClient
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
        self.api_client = None
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
            # Initialize API client for Advance mode
            self.api_client = GemmaAPIClient()
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

    def _listen_for_mode_wake_words(self):
        """Listen for Basic or Advance wake words for mode selection."""
        import scipy.signal
        from backend.utils.config import AUDIO_SETTINGS
        
        dotenv.load_dotenv(dotenv.find_dotenv())
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_ACCESS_KEY not set in environment or .env file.")
        
        # Define mode wake word paths
        basic_keyword_path = str(Path(__file__).parent / "porcupine" / "Basic.ppn")
        advance_keyword_path = str(Path(__file__).parent / "porcupine" / "Advance.ppn")
        
        # Create Porcupine with mode wake words
        porcupine = pvporcupine.create(
            access_key=access_key, 
            keyword_paths=[basic_keyword_path, advance_keyword_path], 
            sensitivities=[0.7, 0.7]
        )
        
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
        
        print(f"\n👂 Listening for mode wake words: 'Basic' or 'Advance'...")
        logger.info(f"Listening for mode wake words with resampling {device_sample_rate}Hz → {porcupine_sample_rate}Hz")
        detected_mode = None
        
        try:
            start_time = time.time()
            timeout = 15.0  # 15 second timeout for mode selection
            
            while not detected_mode and self.running:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.info("Mode selection timeout reached")
                    break
                
                # Read audio at device sample rate
                pcm_data = audio_stream.read(device_frame_length, exception_on_overflow=False)
                pcm_array = struct.unpack_from("h" * device_frame_length, pcm_data)
                
                # Resample to Porcupine's required sample rate
                resampled = scipy.signal.resample(pcm_array, porcupine.frame_length)
                resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
                
                # Process with Porcupine
                result = porcupine.process(resampled_int16)
                if result >= 0:
                    if result == 0:  # Basic.ppn detected
                        print("\n🏠 Wake word 'Basic' detected!")
                        logger.info("Basic wake word detected!")
                        detected_mode = "basic"
                    elif result == 1:  # Advance.ppn detected
                        print("\n🧠 Wake word 'Advance' detected!")
                        logger.info("Advance wake word detected!")
                        detected_mode = "advance"
                    break
                    
        except Exception as e:
            logger.error(f"Error in mode wake word detection: {e}")
        finally:
            audio_stream.close()
            pa.terminate()
            porcupine.delete()
            
        return detected_mode

    def _handle_mode_selection(self):
        """Handle the mode selection process using wake words"""
        max_attempts = 3
        for attempt in range(max_attempts):
            # Ask for mode selection
            mode_prompt = self._get_mode_selection_prompt()
            print(f"🤖 **Quantum Hive:** {mode_prompt}")
            logger.info(f"Asking for mode selection: {mode_prompt}")
            self.tts_engine.speak(mode_prompt)
            
            # Listen for mode wake words
            print("\n🎤 Please say 'Basic' for smart home control or 'Advance' for AI chat...")
            selected_mode = self._listen_for_mode_wake_words()
            
            if selected_mode:
                if selected_mode == "basic":
                    confirmation = "Basic mode activated, Master. Ready to control your smart home devices."
                else:  # advance
                    confirmation = "Advance mode activated, Master. Ready for intelligent conversation."
                
                print(f"🤖 **Mode Selected:** {selected_mode.title()}")
                logger.info(f"Mode selected: {selected_mode}")
                self.tts_engine.speak(confirmation)
                return selected_mode
            else:
                # No wake word detected
                timeout_msg = "I didn't hear your mode selection, Master. Please try again."
                print(f"🤖 **No Selection:** {timeout_msg}")
                self.tts_engine.speak(timeout_msg)
        
        # After max attempts, default to advance mode
        default_msg = "Defaulting to Advance mode, Master."
        print(f"🤖 **Default Mode:** {default_msg}")
        logger.info("Defaulting to advance mode after max attempts")
        self.tts_engine.speak(default_msg)
        return "advance"

    def _handle_basic_mode(self):
        """Handle Basic mode - Smart home device control"""
        print("\n🏠 **Basic Mode Active** - Ready for smart home commands")
        logger.info("Entering Basic mode")
        
        # Listen for basic control commands
        print("🎤 What would you like me to control in your home?")
        basic_command = self.stt_engine.listen_for_speech(timeout=25.0, min_record_time=2.0)
        
        if basic_command:
            print(f"🎤 **Basic Command:** {basic_command}")
            logger.info(f"Basic command: {basic_command}")
            
            # Process basic command
            response = self._process_basic_command(basic_command)
            print(f"🏠 **Basic Response:** {response}")
            logger.info(f"Basic response: {response}")
            self.tts_engine.speak(response)
        else:
            no_command_msg = "I didn't hear a command, Master. Say 'Activate Hive' to try again."
            print(f"🏠 **No Command:** {no_command_msg}")
            self.tts_engine.speak(no_command_msg)

    def _process_basic_command(self, command):
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
        """Handle Advance mode - AI conversation with Gemma API (loops until exit)"""
        print("\n🧠 **Advance Mode Active** - Ready for intelligent conversation")
        logger.info("Entering Advance mode loop (will remain until exit command)")

        while self.running:
            print("\n💬 Please speak your question or request clearly (or say 'exit' to leave)...")
            user_input = self.stt_engine.listen_for_speech(timeout=10.0, min_record_time=2.0)

            if not user_input or not user_input.strip():
                no_input_msg = "I didn't hear anything, Master. Please try again."
                print(f"🧠 **No Input:** {no_input_msg}")
                self.tts_engine.speak(no_input_msg)
                continue  # stay in advance mode

            print(f"\n🎤 **You said:** {user_input}")
            logger.info(f"User said: {user_input}")

            # Handle exit command
            if self._is_exit_command(user_input):
                print(f"\n🛑 **Exit command detected:** {user_input}")
                logger.info("Exit command detected, leaving Advance mode loop")
                self.stop()
                return  # completely stop application

            # Handle memory search command
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
                except Exception as e:
                    logger.error(f"Memory search failed: {e}")
                    response = "Sorry Master, I had trouble searching my memory."
                    self.tts_engine.speak(response)
                # continue listening in Advance mode
                last_user_input = user_input  # update for potential next memory search
                continue

            # Generate AI response via API
            logger.info("Generating AI response via Gemma API...")
            history_pairs = [(msg["user"], msg["ai"]) for msg in self.conversation_history[-6:]]
            ai_response = self.api_client.generate_response(
                user_input,
                system_prompt=self.system_prompt,
                history=history_pairs
            )

            # Debug validation
            print(f"\n🔍 [DEBUG] Raw AI response: '{ai_response}' (length {len(ai_response) if ai_response else 0})")

            if ai_response and ai_response.strip():
                lower_resp = ai_response.lower().strip()
                error_prefixes = (
                    "sorry master, i'm having trouble",
                    "sorry master, i encountered",
                    "sorry master, my response is taking",
                    "sorry master, i'm having connection"
                )
                if any(lower_resp.startswith(p) for p in error_prefixes):
                    # Error-style response, inform user and retry loop
                    error_msg = "I'm having trouble connecting to my advanced AI system. Let's try again."
                    print(f"🧠 **AI Error Response:** {error_msg}")
                    self.tts_engine.speak(error_msg)
                    continue  # stay in advance mode

                # Valid response
                print(f"🤖 **AI Response:** {ai_response}")
                logger.info(f"AI response: {ai_response}")
                self.tts_engine.speak(ai_response)

                # Save conversation
                self.conversation_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "user": user_input,
                    "ai": ai_response,
                    "user_embedding": self._embed_text(user_input),
                    "ai_embedding": self._embed_text(ai_response)
                })
                self._save_history()

                # Update last user input for memory search
                last_user_input = user_input
            else:
                # Empty response from API
                error_msg = "I didn't get a response from the AI. Let's try again."
                print(f"🧠 **Empty AI Response:** {error_msg}")
                self.tts_engine.speak(error_msg)
                # continue loop to listen again

    def _embed_text(self, text):
        """Embed a single text string using the SentenceTransformer."""
        return self.embedder.encode(text, convert_to_tensor=True).tolist()

    def _listen_for_wake_word(self):
        """Listen for the main 'Activate Hive' wake word."""
        import scipy.signal
        from backend.utils.config import AUDIO_SETTINGS
        
        dotenv.load_dotenv(dotenv.find_dotenv())
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_ACCESS_KEY not set in environment or .env file.")
        
        # Use the main activate hive wake word
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
                    if selected_mode == "basic":
                        # Basic mode - Smart home control
                        self._handle_basic_mode()
                    elif selected_mode == "advance":
                        # Advance mode - AI conversation with API
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