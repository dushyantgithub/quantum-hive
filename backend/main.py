"""
Quantum Hive - Main Application
Orchestrates STT → AI → TTS workflow
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.tts.tts_engine import TTSEngine
from backend.ai.gemma_text_engine import TinyLlamaTextAIEngine
from backend.utils.config import LOGGING_SETTINGS, STT_SETTINGS, TTS_SETTINGS, MEMORY_SETTINGS

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

    def _is_memory_search_command(self, text):
        triggers = [
            "what did i say", "what was it", "remind me", "what did we talk about", "what did i mention"
        ]
        return any(trigger in text.lower() for trigger in triggers)

    def start(self):
        logger.info("Starting Quantum Hive...")
        self.running = True
        welcome_message = "Hello! I am Quantum Hive, your AI assistant. I'm ready to help you!"
        logger.info("Quantum Hive is ready!")
        try:
            self.tts_engine.speak(welcome_message)
            last_user_input = None
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
                        if self._is_memory_search_command(user_input) and last_user_input:
                            logger.info("Memory search command detected.")
                            results = self.search_memory(last_user_input, top_n=1)
                            if results:
                                mem = results[0]
                                response = f"You said: {mem['user']}. I replied: {mem['ai']}"
                            else:
                                response = "I don't remember anything relevant from the last week."
                            print(f"🤖 **Memory Search Result:** {response}")
                            self.tts_engine.speak(response)
                            continue
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
                        last_user_input = user_input
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