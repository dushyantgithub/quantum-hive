"""
Text-to-Speech Engine with multiple backend support
"""
import logging
import tempfile
import os
from pathlib import Path
from ..utils.config import TTS_SETTINGS

logger = logging.getLogger(__name__)

class TTSEngine:
    """Base TTS engine class"""
    
    def __init__(self, engine_type=None):
        """Initialize TTS engine"""
        self.engine_type = engine_type or TTS_SETTINGS["engine"]
        self.voice = TTS_SETTINGS["voice"]
        self.rate = TTS_SETTINGS["rate"]
        self.volume = TTS_SETTINGS["volume"]
        
        self._engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the specific TTS engine"""
        if self.engine_type == "espeak":
            self._init_espeak()
        elif self.engine_type == "coqui":
            self._init_coqui()
        elif self.engine_type == "pyttsx3":
            self._init_pyttsx3()
        else:
            raise ValueError(f"Unsupported TTS engine: {self.engine_type}")
    
    def _init_espeak(self):
        """Initialize eSpeak TTS engine"""
        try:
            import subprocess
            # Check if espeak is installed
            result = subprocess.run(["espeak", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("eSpeak TTS engine initialized")
            else:
                raise RuntimeError("eSpeak not found. Please install espeak-ng")
        except FileNotFoundError:
            raise RuntimeError("eSpeak not found. Please install espeak-ng")
    
    def _init_coqui(self):
        """Initialize Coqui TTS engine"""
        try:
            from TTS.api import TTS
            self._engine = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Coqui TTS engine initialized")
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS")
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 TTS engine"""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            
            # Set voice if available
            voices = self._engine.getProperty('voices')
            if voices:
                self._engine.setProperty('voice', voices[0].id)
            
            logger.info("pyttsx3 TTS engine initialized")
        except ImportError:
            raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")
    
    def speak(self, text, save_to_file=None):
        """
        Convert text to speech and play it
        
        Args:
            text (str): Text to convert to speech
            save_to_file (str, optional): Path to save audio file
            
        Returns:
            str: Path to saved audio file if save_to_file is provided
        """
        if not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        try:
            if self.engine_type == "espeak":
                return self._speak_espeak(text, save_to_file)
            elif self.engine_type == "coqui":
                return self._speak_coqui(text, save_to_file)
            elif self.engine_type == "pyttsx3":
                return self._speak_pyttsx3(text, save_to_file)
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return None
    
    def _speak_espeak(self, text, save_to_file=None):
        """Speak using eSpeak"""
        import subprocess
        
        cmd = [
            "espeak",
            "-v", self.voice,
            "-s", str(self.rate),
            "-a", str(int(self.volume * 200)),  # eSpeak volume is 0-200
            text
        ]
        
        if save_to_file:
            cmd.extend(["-w", save_to_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"eSpeak: '{text}'")
                return save_to_file
            else:
                logger.error(f"eSpeak error: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"eSpeak execution error: {e}")
            return None
    
    def _speak_coqui(self, text, save_to_file=None):
        """Speak using Coqui TTS"""
        if not save_to_file:
            save_to_file = tempfile.mktemp(suffix=".wav")
        
        try:
            self._engine.tts_to_file(text=text, file_path=save_to_file)
            logger.info(f"Coqui TTS: '{text}' -> {save_to_file}")
            return save_to_file
        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")
            return None
    
    def _speak_pyttsx3(self, text, save_to_file=None):
        """Speak using pyttsx3"""
        try:
            if save_to_file:
                # pyttsx3 doesn't directly support saving to file
                # We'll use a workaround with temporary file
                temp_file = tempfile.mktemp(suffix=".wav")
                self._engine.save_to_file(text, temp_file)
                self._engine.runAndWait()
                
                # Move to desired location
                import shutil
                shutil.move(temp_file, save_to_file)
                logger.info(f"pyttsx3: '{text}' -> {save_to_file}")
                return save_to_file
            else:
                self._engine.say(text)
                self._engine.runAndWait()
                logger.info(f"pyttsx3: '{text}'")
                return None
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return None
    
    def play_audio_file(self, audio_file_path):
        """
        Play an audio file
        
        Args:
            audio_file_path (str): Path to audio file
        """
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.quit()
            logger.info(f"Played audio file: {audio_file_path}")
            
        except ImportError:
            logger.warning("pygame not available, using system audio player")
            self._play_with_system_player(audio_file_path)
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
    
    def _play_with_system_player(self, audio_file_path):
        """Play audio using system default player"""
        import subprocess
        import platform
        
        system = platform.system().lower()
        
        try:
            if system == "darwin":  # macOS
                subprocess.run(["afplay", audio_file_path])
            elif system == "linux":
                subprocess.run(["aplay", audio_file_path])
            elif system == "windows":
                subprocess.run(["start", audio_file_path], shell=True)
            else:
                logger.warning(f"Unknown system: {system}, cannot play audio")
        except Exception as e:
            logger.error(f"Error playing with system player: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self._engine and hasattr(self._engine, 'cleanup'):
            self._engine.cleanup()
        logger.info("TTS engine cleaned up")

# Convenience function for quick text-to-speech
def speak_text(text, engine_type=None, save_to_file=None):
    """
    Quick function to convert text to speech
    
    Args:
        text (str): Text to convert
        engine_type (str): TTS engine to use
        save_to_file (str): Optional file path to save audio
        
    Returns:
        str: Path to saved audio file if save_to_file is provided
    """
    engine = TTSEngine(engine_type)
    try:
        return engine.speak(text, save_to_file)
    finally:
        engine.cleanup()

if __name__ == "__main__":
    # Test the TTS engine
    logging.basicConfig(level=logging.INFO)
    print("Testing TTS Engine...")
    
    test_text = "Hello, I am Quantum Hive, your AI assistant!"
    
    # Test with different engines
    engines = ["espeak", "pyttsx3"]  # Skip coqui for quick test
    
    for engine_type in engines:
        try:
            print(f"\nTesting {engine_type}...")
            result = speak_text(test_text, engine_type)
            if result:
                print(f"Audio saved to: {result}")
            else:
                print("TTS completed successfully")
        except Exception as e:
            print(f"Error with {engine_type}: {e}") 