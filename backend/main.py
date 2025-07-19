"""
Quantum Hive - Main Application
Orchestrates STT → AI → TTS workflow
"""
import logging
import time
import signal
import sys
from pathlib import Path

# Import our modules
from stt.vosk_engine import VoskSTTEngine
from tts.tts_engine import TTSEngine
from ai.local_llm import get_ai_engine
from utils.config import LOGGING_SETTINGS, STT_SETTINGS, TTS_SETTINGS, AI_SETTINGS

# Configure logging
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
        """Initialize Quantum Hive"""
        self.running = False
        self.stt_engine = None
        self.tts_engine = None
        self.ai_engine = None
        
        # System prompt for the AI
        self.system_prompt = """You are Quantum Hive, a helpful AI assistant running on a Raspberry Pi. 
        You are designed to be conversational, helpful, and concise in your responses. 
        You can help with various tasks and answer questions. Keep responses under 100 words unless asked for more detail."""
        
        self._initialize_components()
        self._setup_signal_handlers()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Quantum Hive components...")
            
            # Initialize STT engine
            logger.info("Initializing STT engine...")
            self.stt_engine = VoskSTTEngine()
            
            # Initialize TTS engine
            logger.info("Initializing TTS engine...")
            self.tts_engine = TTSEngine()
            
            # Initialize AI engine
            logger.info("Initializing AI engine...")
            self.ai_engine = get_ai_engine()
            
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self):
        """Start the Quantum Hive application"""
        logger.info("Starting Quantum Hive...")
        self.running = True
        
        # Welcome message
        welcome_message = "Hello! I am Quantum Hive, your AI assistant. I'm ready to help you!"
        logger.info("Quantum Hive is ready!")
        
        try:
            # Speak welcome message
            self.tts_engine.speak(welcome_message)
            
            # Main loop
            while self.running:
                try:
                    # Listen for speech
                    logger.info("Listening for speech...")
                    user_input = self.stt_engine.listen_for_speech(timeout=STT_SETTINGS["timeout"])
                    
                    if user_input:
                        logger.info(f"User said: {user_input}")
                        
                        # Check for exit commands
                        if self._is_exit_command(user_input):
                            logger.info("Exit command detected")
                            break
                        
                        # Generate AI response
                        logger.info("Generating AI response...")
                        ai_response = self.ai_engine.generate_response(
                            user_input, 
                            system_prompt=self.system_prompt
                        )
                        
                        logger.info(f"AI response: {ai_response}")
                        
                        # Speak the response
                        logger.info("Speaking response...")
                        self.tts_engine.speak(ai_response)
                        
                    else:
                        logger.debug("No speech detected")
                        
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    # Try to speak error message
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
        """Check if the user wants to exit"""
        exit_commands = [
            "exit", "quit", "stop", "goodbye", "bye", "shutdown", 
            "turn off", "power off", "end session"
        ]
        return any(cmd in text.lower() for cmd in exit_commands)
    
    def stop(self):
        """Stop the Quantum Hive application"""
        logger.info("Stopping Quantum Hive...")
        self.running = False
        
        # Cleanup components
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
    """Main entry point"""
    print("🧠 Quantum Hive - AI Assistant")
    print("=" * 40)
    
    try:
        # Create and start Quantum Hive
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