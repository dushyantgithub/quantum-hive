#!/usr/bin/env python3
"""
Test script for Advance Mode API flow
Simulates the advance mode without audio components
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

def test_advance_mode():
    print("🧠 Testing Advance Mode API Flow")
    print("=" * 60)
    
    # Set the correct endpoint
    os.environ["LOCAL_LLM_ENDPOINT"] = "http://dushyant-pc.tailde7d3d.ts.net:11434/api/generate"
    
    # Initialize client
    print("\n1. Initializing API Client...")
    try:
        client = GemmaAPIClient()
        print(f"   ✅ Client initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize client: {e}")
        return
    
    # Test system prompt
    system_prompt = (
        "You are Quantum Hive, a helpful AI assistant running on a Raspberry Pi. "
        "You are designed to be conversational, helpful, and concise in your responses. "
        "You can help with various tasks and answer questions. Keep responses under 100 words unless asked for more detail."
    )
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a joke",
        "Thank you for your help"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        print("-" * 40)
        
        try:
            response = client.generate_response(
                prompt,
                system_prompt=system_prompt,
                history=[]
            )
            
            print(f"\n📋 Final Response: '{response}'")
            print(f"   Response type: {type(response)}")
            print(f"   Response length: {len(response)}")
            
            # Simulate the validation logic from main.py
            print(f"\n🔍 Validation Check:")
            if response and response.strip():
                lower_response = response.lower().strip()
                is_error_response = (
                    lower_response.startswith("sorry master, i'm having trouble") or
                    lower_response.startswith("sorry master, i encountered") or
                    lower_response.startswith("sorry master, my response is taking") or
                    lower_response.startswith("sorry master, i'm having connection")
                )
                
                if not is_error_response:
                    print(f"   ✅ Response would be ACCEPTED by main app")
                else:
                    print(f"   ❌ Response would be REJECTED by main app (error message)")
            else:
                print(f"   ❌ Response would be REJECTED by main app (empty)")
                
        except Exception as e:
            print(f"   💥 Exception during API call: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏥 Health Check:")
    try:
        is_healthy = client.is_healthy()
        print(f"   API Health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_advance_mode() 