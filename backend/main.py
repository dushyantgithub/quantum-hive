"""
Quantum Hive - Main Application
Orchestrates STT → AI → TTS workflow
"""
import logging
import time
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.tts.tts_engine import TTSEngine
from backend.ai.gemma_text_engine import TinyLlamaTextAIEngine
from backend.utils.config import LOGGING_SETTINGS, STT_SETTINGS, TTS_SETTINGS

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
            logger.info("Initializing TinyLlama AI engine...")
            self.ai_engine = TinyLlamaTextAIEngine()
            logger.info("All components initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def start(self):
        logger.info("Starting Quantum Hive...")
        self.running = True
        welcome_message = "Hello! I am Quantum Hive, your AI assistant. I'm ready to help you!"
        logger.info("Quantum Hive is ready!")
        try:
            self.tts_engine.speak(welcome_message)
            while self.running:
                try:
                    logger.info("Listening for speech...")
                    user_input = self.stt_engine.listen_for_speech(timeout=30.0, min_record_time=3.0)
                    if user_input:
                        print(f"\n🎤 **You said:** {user_input}")
                        logger.info(f"User said: {user_input}")
                        if self._is_exit_command(user_input):
                            print(f"\n🛑 **Exit command detected:** {user_input}")
                            logger.info("Exit command detected")
                            break
                        logger.info("Generating AI response...")
                        ai_response = self.ai_engine.generate_response(
                            user_input,
                            system_prompt=self.system_prompt
                        )
                        print(f"🤖 **AI Response:** {ai_response}")
                        logger.info(f"AI response: {ai_response}")
                        logger.info("Speaking response...")
                        self.tts_engine.speak(ai_response)
                        print(f"\n🎧 **Listening for speech...**")
                    else:
                        logger.debug("No speech detected")
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

    def _is_exit_command(self, text):
        exit_commands = [
            "exit", "quit", "stop", "goodbye", "bye", "shutdown", 
            "turn off", "power off", "end session"
        ]
        return any(cmd in text.lower() for cmd in exit_commands)

    def stop(self):
        logger.info("Stopping Quantum Hive...")
        self.running = False
        try:
            if self.stt_engine:
                self.stt_engine.cleanup()
            if self.tts_engine:
                self.tts_engine.cleanup()
            if self.ai_engine:
                self.ai_engine.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        logger.info("Quantum Hive stopped")

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