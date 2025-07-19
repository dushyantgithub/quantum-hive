"""
Vosk-based Speech-to-Text Engine
"""
import pyaudio
import json
import logging
from vosk import Model, KaldiRecognizer
from ..utils.config import STT_SETTINGS, AUDIO_SETTINGS

logger = logging.getLogger(__name__)

class VoskSTTEngine:
    """Vosk-based Speech-to-Text engine"""
    
    def __init__(self, model_path=None):
        """Initialize Vosk STT engine"""
        self.model_path = model_path or STT_SETTINGS["model_path"]
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.channels = AUDIO_SETTINGS["channels"]
        
        try:
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.audio = pyaudio.PyAudio()
            logger.info(f"Vosk STT engine initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Vosk STT engine: {e}")
            raise
    
    def listen_for_speech(self, timeout=5.0):
        """
        Listen for speech input and return transcribed text
        
        Args:
            timeout (float): Maximum time to listen in seconds
            
        Returns:
            str: Transcribed text or None if no speech detected
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Listening for speech...")
            
            # Listen for audio input
            audio_data = b""
            silence_frames = 0
            max_silence_frames = int(self.sample_rate / self.chunk_size * 1.0)  # 1 second of silence
            
            while True:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data += data
                    
                    # Check if Vosk has a result
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text", "").strip():
                            logger.info(f"Speech detected: {result['text']}")
                            stream.close()
                            return result["text"]
                    
                    # Check for silence
                    if self._is_silence(data):
                        silence_frames += 1
                        if silence_frames > max_silence_frames and audio_data:
                            # Process remaining audio
                            if self.recognizer.AcceptWaveform(audio_data):
                                result = json.loads(self.recognizer.Result())
                                if result.get("text", "").strip():
                                    logger.info(f"Speech detected: {result['text']}")
                                    stream.close()
                                    return result["text"]
                            break
                    else:
                        silence_frames = 0
                        
                except KeyboardInterrupt:
                    logger.info("Listening interrupted by user")
                    break
            
            stream.close()
            return None
            
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            return None
    
    def _is_silence(self, audio_chunk, threshold=500):
        """Check if audio chunk is silence"""
        import struct
        # Convert bytes to 16-bit integers
        audio_data = struct.unpack(f"{len(audio_chunk)//2}h", audio_chunk)
        # Calculate RMS (Root Mean Square) of audio data
        rms = (sum(x*x for x in audio_data) / len(audio_data)) ** 0.5
        return rms < threshold
    
    def transcribe_audio_file(self, audio_file_path):
        """
        Transcribe audio from a file
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            import wave
            
            with wave.open(audio_file_path, 'rb') as wf:
                # Read audio data
                audio_data = wf.readframes(wf.getnframes())
                
                # Process with Vosk
                if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.Result())
                    return result.get("text", "")
                else:
                    # Get partial result
                    result = json.loads(self.recognizer.PartialResult())
                    return result.get("partial", "")
                    
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.audio.terminate()
            logger.info("Vosk STT engine cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Convenience function for quick speech recognition
def listen_and_transcribe(timeout=5.0):
    """
    Quick function to listen and transcribe speech
    
    Args:
        timeout (float): Maximum time to listen
        
    Returns:
        str: Transcribed text or None
    """
    engine = VoskSTTEngine()
    try:
        return engine.listen_for_speech(timeout)
    finally:
        engine.cleanup()

if __name__ == "__main__":
    # Test the STT engine
    logging.basicConfig(level=logging.INFO)
    print("Testing Vosk STT Engine...")
    print("Speak something (press Ctrl+C to stop)...")
    
    try:
        text = listen_and_transcribe()
        if text:
            print(f"Transcribed: {text}")
        else:
            print("No speech detected")
    except KeyboardInterrupt:
        print("\nTest interrupted") 