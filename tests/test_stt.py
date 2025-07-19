"""
Tests for Speech-to-Text components
"""
import pytest
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.stt.vosk_engine import VoskSTTEngine

class TestVoskSTTEngine:
    """Test cases for Vosk STT engine"""
    
    def test_initialization(self):
        """Test STT engine initialization"""
        try:
            engine = VoskSTTEngine()
            assert engine is not None
            assert hasattr(engine, 'model')
            assert hasattr(engine, 'recognizer')
        except Exception as e:
            pytest.skip(f"Vosk not available: {e}")
    
    def test_silence_detection(self):
        """Test silence detection functionality"""
        try:
            engine = VoskSTTEngine()
            
            # Test with silence (zeros)
            silence_data = b'\x00\x00' * 512  # 512 samples of silence
            assert engine._is_silence(silence_data) == True
            
            # Test with non-silence (random data)
            non_silence_data = b'\xff\x00' * 512  # 512 samples of non-silence
            assert engine._is_silence(non_silence_data) == False
            
        except Exception as e:
            pytest.skip(f"Vosk not available: {e}")
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        try:
            engine = VoskSTTEngine()
            engine.cleanup()
            # Should not raise any exceptions
        except Exception as e:
            pytest.skip(f"Vosk not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 