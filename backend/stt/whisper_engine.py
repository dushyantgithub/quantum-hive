"""
Whisper-based Speech-to-Text Engine
Provides much better accuracy than Vosk while remaining lightweight
"""
import logging
import tempfile
import os
import time
import pyaudio
import wave
import whisper
import numpy as np
from pathlib import Path
from ..utils.config import STT_SETTINGS, AUDIO_SETTINGS

logger = logging.getLogger(__name__)

class WhisperSTTEngine:
    """Whisper-based STT engine with real-time microphone input"""
    
    def __init__(self, model_size="base"):
        """
        Initialize Whisper STT engine
        
        Args:
            model_size (str): Model size - "tiny", "base", "small", "medium", "large"
                             For Raspberry Pi, use "tiny" or "base" for best performance
        """
        self.model_size = model_size
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.channels = AUDIO_SETTINGS["channels"]
        self.format = pyaudio.paInt16
        
        # Audio settings
        self.silence_threshold = STT_SETTINGS["silence_threshold"]
        self.silence_duration = STT_SETTINGS["silence_duration"]
        self.min_audio_duration = STT_SETTINGS["min_audio_duration"]
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Load Whisper model
        self._load_model()
        
        logger.info(f"Whisper STT engine initialized with model: {model_size}")
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info(f"Whisper model loaded successfully: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _record_audio(self, timeout=None):
        """
        Record audio from microphone
        
        Args:
            timeout (float): Maximum recording time in seconds
            
        Returns:
            bytes: Audio data
        """
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silent_chunks = 0
            silent_threshold = int(self.silence_duration * self.sample_rate / self.chunk_size)
            min_chunks = int(self.min_audio_duration * self.sample_rate / self.chunk_size)
            
            logger.info("Listening for speech...")
            
            start_time = time.time()
            
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.info("Recording timeout reached")
                    break
                
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy array for silence detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))
                
                # Check for silence
                if volume < self.silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                # Stop if silence detected and minimum duration met
                if silent_chunks >= silent_threshold and len(frames) >= min_chunks:
                    logger.info("Silence detected, stopping recording")
                    break
            
            # Close stream
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            return b''.join(frames)
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None
            raise
    
    def _save_audio_temp(self, audio_data):
        """
        Save audio data to temporary file
        
        Args:
            audio_data (bytes): Audio data
            
        Returns:
            str: Path to temporary audio file
        """
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save as WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving audio to temp file: {e}")
            raise
    
    def _transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=False  # Disable for better compatibility
            )
            
            text = result["text"].strip()
            logger.info(f"Whisper transcription: {text}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
    
    def listen_for_speech(self, timeout=None):
        """
        Listen for speech and return transcribed text
        
        Args:
            timeout (float): Maximum listening time in seconds
            
        Returns:
            str: Transcribed text or None if no speech detected
        """
        try:
            # Record audio
            audio_data = self._record_audio(timeout)
            
            if not audio_data:
                return None
            
            # Save to temporary file
            temp_path = self._save_audio_temp(audio_data)
            
            # Transcribe
            text = self._transcribe_audio(temp_path)
            
            if text and len(text.strip()) > 0:
                logger.info(f"Speech detected: {text}")
                return text
            else:
                logger.info("No speech detected")
                return None
                
        except Exception as e:
            logger.error(f"Error in listen_for_speech: {e}")
            return None
    
    def transcribe_file(self, audio_file_path):
        """
        Transcribe an audio file
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            if not Path(audio_file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            text = self._transcribe_audio(audio_file_path)
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            logger.info("Whisper STT engine cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Test the Whisper STT engine
    logging.basicConfig(level=logging.INFO)
    print("Testing Whisper STT Engine...")
    
    # Initialize with base model (good balance of accuracy and speed)
    stt_engine = WhisperSTTEngine(model_size="base")
    
    try:
        print("Speak something (press Ctrl+C to stop)...")
        while True:
            text = stt_engine.listen_for_speech(timeout=10)
            if text:
                print(f"You said: {text}")
            else:
                print("No speech detected")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stt_engine.cleanup() 