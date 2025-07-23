#!/usr/bin/env python3
"""
Test script for GemmaAPIClient
Tests both normal API calls and fallback responses
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from ai.gemma_api_client import GemmaAPIClient
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_api_client():
    print("🧪 Testing GemmaAPIClient")
    print("=" * 50)
    
    # Test 1: Normal mode (with current API issues)
    print("\n1. Testing Normal Mode:")
    try:
        client = GemmaAPIClient()
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Thank you for your help"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = client.generate_response(prompt)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Error in normal mode: {e}")
    
    # Test 2: Test mode (fallback responses)
    print("\n\n2. Testing Fallback Mode:")
    try:
        os.environ["TEST_MODE"] = "true"
        client_test = GemmaAPIClient()
        
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?", 
            "Thank you for your help",
            "How do you do?",
            "What can you tell me about AI?"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = client_test.generate_response(prompt)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Error in test mode: {e}")
    
    # Test 3: Health check
    print("\n\n3. Testing Health Check:")
    try:
        print(f"Normal mode healthy: {client.is_healthy()}")
        print(f"Test mode healthy: {client_test.is_healthy()}")
    except Exception as e:
        print(f"Error in health check: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_api_client() 