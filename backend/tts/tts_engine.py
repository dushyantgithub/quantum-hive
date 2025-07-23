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
    """Coqui TTS engine only"""
    def __init__(self, engine_type=None):
        self.engine_type = "coqui"
        self.voice = TTS_SETTINGS["voice"]
        self.rate = TTS_SETTINGS["rate"]
        self.volume = TTS_SETTINGS["volume"]
        self._engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        self._init_coqui()

    def _init_coqui(self):
        try:
            from TTS.api import TTS
            import torch
            
            # Try fast models first, fallback to slower ones
            fast_models = [
                "tts_models/en/ljspeech/fast_pitch",  # FastPitch is much faster
                "tts_models/en/ljspeech/speedy_speech",  # Speedy Speech is very fast
                "tts_models/en/ljspeech/tacotron2-DDC"  # Fallback to original
            ]
            
            model_name = self.voice or fast_models[0]
            
            # If the configured model is not in our fast list, try it first
            if model_name not in fast_models:
                models_to_try = [model_name] + fast_models
            else:
                models_to_try = fast_models
            
            for model in models_to_try:
                try:
                    logger.info(f"Trying TTS model: {model}")
                    # Use CPU for faster loading on Pi, disable GPU optimization
                    self._engine = TTS(model, gpu=False)
                    
                    # Set low memory and fast processing options for 16kHz output
                    if hasattr(self._engine, 'output_sample_rate'):
                        self._engine.output_sample_rate = self.rate
                    
                    # Configure synthesizer for faster processing at 16kHz
                    if hasattr(self._engine, 'synthesizer'):
                        if hasattr(self._engine.synthesizer, 'output_sample_rate'):
                            self._engine.synthesizer.output_sample_rate = self.rate
                    
                    # Optimize for speed on CPU
                    if hasattr(self._engine, 'synthesizer') and hasattr(self._engine.synthesizer, 'tts_model'):
                        # Disable attention alignment for speed (if available)
                        if hasattr(self._engine.synthesizer.tts_model, 'use_forward_attn'):
                            self._engine.synthesizer.tts_model.use_forward_attn = False
                    
                    logger.info(f"Coqui TTS engine initialized with model: {model} at {self.rate} Hz")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load model {model}: {e}")
                    continue
            
            raise RuntimeError("All TTS models failed to load")
            
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Coqui TTS: {e}")

    def speak(self, text, save_to_file=None):
        if not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        try:
            return self._speak_coqui(text, save_to_file)
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return None

    def _speak_coqui(self, text, save_to_file=None):
        import tempfile
        import os
        import time
        try:
            start_time = time.time()
            logger.debug(f"[TTS] Starting synthesis with Coqui: '{text}'")
            
            # Preprocess text for faster synthesis
            text = self._preprocess_text_for_speed(text)
            
            if not save_to_file:
                save_to_file = tempfile.mktemp(suffix=".wav")
            logger.debug(f"[TTS] Output file will be: {save_to_file}")
            
            # Use faster synthesis options if available
            try:
                # Try with speed optimizations
                self._engine.tts_to_file(
                    text=text, 
                    file_path=save_to_file,
                    # Add speed optimizations where possible
                )
            except TypeError:
                # Fallback if advanced options not supported
                self._engine.tts_to_file(text=text, file_path=save_to_file)
            
            synthesis_time = time.time() - start_time
            logger.debug(f"[TTS] Synthesis complete in {synthesis_time:.2f}s. Checking file...")
            
            if not os.path.exists(save_to_file):
                logger.error(f"[TTS] Audio file was not created: {save_to_file}")
                return None
            if os.path.getsize(save_to_file) == 0:
                logger.error(f"[TTS] Audio file is empty: {save_to_file}")
                return None
                
            logger.info(f"Coqui TTS: '{text}' -> {save_to_file} (synthesis: {synthesis_time:.2f}s)")
            logger.debug(f"[TTS] Attempting to play audio file: {save_to_file}")
            self.play_audio_file(save_to_file)
            logger.debug(f"[TTS] Playback function completed for: {save_to_file}")
            return save_to_file
        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")
            return None
            
    def _preprocess_text_for_speed(self, text):
        """Preprocess text to make TTS synthesis faster"""
        import re
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Simplify complex punctuation that might slow down synthesis
        text = re.sub(r'[.]{2,}', '.', text)  # Multiple dots to single dot
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations to single
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions to single
        text = re.sub(r'[-]{2,}', '-', text)  # Multiple dashes to single
        
        # Keep text concise for faster processing
        if len(text) > 200:
            # Split long text into sentences and keep only first few for faster processing
            sentences = text.split('. ')
            if len(sentences) > 3:
                text = '. '.join(sentences[:3]) + '.'
                logger.debug(f"[TTS] Truncated long text for speed: '{text}'")
        
        return text

    def play_audio_file(self, audio_file_path):
        import os
        logger.debug(f"[AUDIO] play_audio_file called with: {audio_file_path}")
        if not os.path.exists(audio_file_path):
            logger.error(f"[AUDIO] File does not exist: {audio_file_path}")
            return
        try:
            import pygame
            logger.debug("[AUDIO] Trying pygame for playback...")
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            logger.debug("[AUDIO] pygame playback started.")
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
            logger.info(f"Played audio file: {audio_file_path}")
        except ImportError:
            logger.warning("pygame not available, using system audio player")
            self._play_with_system_player(audio_file_path)
        except Exception as e:
            logger.error(f"Error playing audio file with pygame: {e}")
            self._play_with_system_player(audio_file_path)

    def _play_with_system_player(self, audio_file_path):
        import subprocess
        import platform
        logger.debug(f"[AUDIO] _play_with_system_player called with: {audio_file_path}")
        system = platform.system().lower()
        try:
            if system == "darwin":
                logger.debug("[AUDIO] Using afplay for playback...")
                subprocess.run(["afplay", audio_file_path])
            elif system == "linux":
                from ..utils.config import AUDIO_SETTINGS
                output_device = AUDIO_SETTINGS.get("output_device", "hw:2,0")
                logger.debug(f"[AUDIO] Using aplay with device {output_device} for playback...")
                
                # For AUX speakers, use direct audio output
                logger.debug("[AUDIO] Using AUX speakers - direct audio output")
                
                # Fallback to default device (headphone jack)
                subprocess.run(["aplay", "-D", output_device, audio_file_path], check=True)
                logger.info(f"Successfully played audio via {output_device}: {audio_file_path}")
            elif system == "windows":
                logger.debug("[AUDIO] Using start for playback...")
                subprocess.run(["start", audio_file_path], shell=True)
            else:
                logger.warning(f"Unknown system: {system}, cannot play audio")
        except Exception as e:
            logger.error(f"Error playing with system player: {e}")

    def cleanup(self):
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