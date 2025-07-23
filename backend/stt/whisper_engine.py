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

try:
    from scipy.signal import resample
    SCIPY_AVAILABLE = True
except ImportError:
    try:
        import resampy
        RESAMPY_AVAILABLE = True
        SCIPY_AVAILABLE = False
    except ImportError:
        SCIPY_AVAILABLE = False
        RESAMPY_AVAILABLE = False
        print("Warning: Neither scipy nor resampy available. Real-time downsampling disabled.")

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
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]  # Input sample rate (44100Hz)
        self.target_sample_rate = STT_SETTINGS.get("downsample_target", 16000)  # Target sample rate (16000Hz)
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.channels = AUDIO_SETTINGS["channels"]
        self.format = pyaudio.paInt16
        
        # Audio settings
        self.silence_threshold = STT_SETTINGS["silence_threshold"]
        self.silence_duration = STT_SETTINGS["silence_duration"]
        self.min_audio_duration = STT_SETTINGS["min_audio_duration"]
        
        # Calculate downsampling ratio
        self.downsample_ratio = self.sample_rate / self.target_sample_rate
        self.target_chunk_size = int(self.chunk_size / self.downsample_ratio)
        
        logger.info(f"Audio downsampling: {self.sample_rate}Hz -> {self.target_sample_rate}Hz (ratio: {self.downsample_ratio:.2f})")
        
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
    
    def _downsample_audio(self, audio_data):
        """
        Downsample audio data from input sample rate to target sample rate
        
        Args:
            audio_data (np.ndarray): Audio data at input sample rate
            
        Returns:
            np.ndarray: Downsampled audio data at target sample rate
        """
        if self.sample_rate == self.target_sample_rate:
            return audio_data
        
        try:
            if SCIPY_AVAILABLE:
                # Use scipy for downsampling
                target_length = int(len(audio_data) / self.downsample_ratio)
                downsampled = resample(audio_data, target_length)
                return downsampled.astype(np.int16)
            elif RESAMPY_AVAILABLE:
                # Use resampy for downsampling
                downsampled = resampy.resample(audio_data.astype(np.float32), 
                                             self.sample_rate, 
                                             self.target_sample_rate)
                return (downsampled * 32767).astype(np.int16)
            else:
                # Fallback: simple decimation (not ideal but functional)
                step = int(self.downsample_ratio)
                return audio_data[::step]
        except Exception as e:
            logger.warning(f"Downsampling failed: {e}, using original audio")
            return audio_data
    
    def _record_audio(self, timeout=None, min_record_time=3.0):
        """
        Record audio from microphone
        
        Args:
            timeout (float): Maximum recording time in seconds
            min_record_time (float): Minimum recording time in seconds (default: 3.0)
            
        Returns:
            bytes: Audio data
        """
        try:
            # Open audio stream
            input_device = AUDIO_SETTINGS.get("input_device")
            open_kwargs = dict(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            if input_device is not None:
                if isinstance(input_device, int):
                    open_kwargs["input_device_index"] = input_device
                else:
                    logger.warning(f"Input device should be an integer index, got: {input_device}")
            print(f"[DEBUG] Opening PyAudio stream with: {open_kwargs}")
            self.stream = self.audio.open(**open_kwargs)
            
            frames = []
            downsampled_frames = []  # Store downsampled audio for processing
            silent_chunks = 0
            silent_threshold = int(self.silence_duration * self.sample_rate / self.chunk_size)
            min_chunks = int(max(self.min_audio_duration, min_record_time) * self.sample_rate / self.chunk_size)
            
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
                
                # Convert to numpy array and downsample for processing
                audio_data = np.frombuffer(data, dtype=np.int16)
                downsampled_audio = self._downsample_audio(audio_data)
                downsampled_frames.append(downsampled_audio.tobytes())
                
                # Use downsampled audio for silence detection (more accurate)
                volume = np.sqrt(np.mean(downsampled_audio**2))
                
                # Check for silence
                if volume < self.silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                elapsed_time = time.time() - start_time
                
                # Stop if silence detected and minimum duration met
                if silent_chunks >= silent_threshold and len(frames) >= min_chunks:
                    logger.info(f"Silence detected after {elapsed_time:.1f}s, stopping recording")
                    break
            
            # Close stream
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            # Return downsampled audio for better Whisper processing
            if downsampled_frames:
                logger.info(f"Recorded {len(downsampled_frames)} downsampled chunks at {self.target_sample_rate}Hz")
                return b''.join(downsampled_frames)
            else:
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
            
            # Save as WAV file with target sample rate
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.target_sample_rate)  # Use downsampled rate
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
    
    def listen_for_speech(self, timeout=None, min_record_time=3.0):
        """
        Listen for speech and return transcribed text
        
        Args:
            timeout (float): Maximum listening time in seconds
            min_record_time (float): Minimum recording time in seconds (default: 3.0)
            
        Returns:
            str: Transcribed text or None if no speech detected
        """
        try:
            # Record audio
            audio_data = self._record_audio(timeout, min_record_time)
            
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
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    print("Testing Whisper STT Engine...")
    
    # Initialize with base model (good balance of accuracy and speed)
    stt_engine = WhisperSTTEngine(model_size="base")
    
    try:
        print("Speak something (press Ctrl+C to stop)...")
        while True:
            text = stt_engine.listen_for_speech(timeout=30, min_record_time=3.0)
            if text:
                print(f"You said: {text}")
            else:
                print("No speech detected")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stt_engine.cleanup() 