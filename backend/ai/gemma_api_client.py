"""
Gemma API Client
Connects to gemma3:12b model running on Ollama server
"""
import logging
import requests
import json
import os
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class GemmaAPIClient:
    """API client for gemma3:12b model via Ollama server"""
    
    def __init__(self):
        """Initialize Gemma API client"""
        raw_endpoint = os.getenv("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
        self.base_url = self._normalize_endpoint(raw_endpoint)
        self.model = "gemma3:12b"
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        self.fallback_responses = [
            "I understand your question. Let me think about that for a moment.",
            "That's an interesting point. Could you tell me more about what you're looking for?",
            "I'm processing your request. Can you provide a bit more context?",
            "Thank you for your question. I'm here to help with whatever you need.",
            "I appreciate you asking. What specific aspect would you like me to focus on?"
        ]
        
        if self.test_mode:
            logger.info("GemmaAPIClient initialized in TEST MODE - using fallback responses")
        else:
            # Test connection
            self._test_connection()
            logger.info(f"Gemma API client initialized (server: {self.base_url})")

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Ensure the endpoint has a scheme and includes /api/generate"""
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = "http://" + endpoint
        # If user supplied only host:port without path, append the standard path
        if endpoint.rstrip("/").endswith(":11434"):
            endpoint = endpoint.rstrip("/") + "/api/generate"
        # If user supplied host without port/path, add default port+path
        if endpoint == "http://127.0.0.1" or endpoint.endswith(".ts.net"):
            if ":" not in endpoint.split("//",1)[1]:
                endpoint = endpoint.rstrip("/") + ":11434/api/generate"
        return endpoint

    def _test_connection(self):
        """Test connection to the API server by generating a short response"""
        try:
            payload = {
                "model": self.model,
                "prompt": "ping",
                "stream": False,
                "options": {"num_thread": 1, "num_ctx": 32}
            }
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    logger.error(f"API server returned error: {result['error']}")
                    logger.warning("API has errors but server is reachable")
                elif "response" in result:
                    logger.info("Successfully connected to Ollama API server")
                else:
                    logger.warning("Unexpected response format from API server")
            else:
                logger.warning(f"API server responded with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API server: {e}")
            logger.error("Make sure your Ollama server is running and accessible")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers (no auth needed for Ollama)"""
        return {"Content-Type": "application/json"}

    def _get_fallback_response(self, user_input: str) -> str:
        """Generate a fallback response when API fails"""
        import random
        base_response = random.choice(self.fallback_responses)
        
        # Add some context-aware responses
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Quantum Hive, your AI assistant. How can I help you today?"
        elif any(word in user_lower for word in ["how are you", "how do you do"]):
            return "I'm doing well, thank you for asking! I'm here and ready to assist you."
        elif any(word in user_lower for word in ["what", "who", "where", "when", "why", "how"]):
            return "That's a great question! I'd be happy to help you with that information."
        elif any(word in user_lower for word in ["thank", "thanks"]):
            return "You're very welcome! I'm glad I could help."
        else:
            return base_response

    def generate_response(self, user_input: str, system_prompt: Optional[str] = None, 
                         history: Optional[List[tuple]] = None) -> str:
        """
        Generate response from gemma3:12b model via Ollama
        """
        if self.test_mode:
            logger.info("Using fallback response (test mode)")
            return self._get_fallback_response(user_input)
            
        try:
            # Compose prompt with optional system prompt and history
            prompt = ""
            if system_prompt:
                prompt += f"[System]\n{system_prompt}\n"
            if history:
                for user_msg, ai_msg in history[-3:]:
                    prompt += f"[User]\n{user_msg}\n[AI]\n{ai_msg}\n"
            prompt += f"[User]\n{user_input}\n[AI]"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_thread": 2, "num_ctx": 512}
            }
            
            # Debug logging
            print(f"\n🔍 [DEBUG] API Request Details:")
            print(f"   URL: {self.base_url}")
            print(f"   Model: {self.model}")
            print(f"   Prompt length: {len(prompt)} chars")
            print(f"   Payload: {json.dumps(payload, indent=2)}")
            logger.info(f"Making API request to {self.base_url}")
            
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            # Debug response
            print(f"\n📡 [DEBUG] API Response Details:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {response_time:.2f}s")
            print(f"   Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"   Response JSON: {json.dumps(result, indent=2)}")
                    
                    # Check for API errors
                    if "error" in result:
                        error_msg = result['error']
                        print(f"❌ [ERROR] API returned error: {error_msg}")
                        logger.error(f"API returned error: {error_msg}")
                        logger.info("Using fallback response due to API error")
                        return self._get_fallback_response(user_input)
                    
                    ai_response = result.get("response", "")
                    if ai_response and ai_response.strip():
                        print(f"✅ [SUCCESS] Got AI response: {ai_response[:100]}...")
                        logger.info(f"Ollama API response received in {response_time:.2f}s")
                        return ai_response.strip()
                    else:
                        print(f"⚠️ [WARNING] API returned empty response")
                        logger.warning("API returned empty response")
                        return self._get_fallback_response(user_input)
                        
                except json.JSONDecodeError as e:
                    print(f"❌ [ERROR] Failed to parse JSON response: {e}")
                    print(f"   Raw response: {response.text}")
                    logger.error(f"Failed to parse JSON response: {e}")
                    return self._get_fallback_response(user_input)
                    
            else:
                print(f"❌ [ERROR] HTTP {response.status_code}: {response.text}")
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return self._get_fallback_response(user_input)
                
        except requests.exceptions.Timeout:
            print(f"⏰ [TIMEOUT] API request timed out after 30 seconds")
            logger.error("API request timed out")
            return self._get_fallback_response(user_input)
            
        except requests.exceptions.RequestException as e:
            print(f"🌐 [CONNECTION ERROR] API request failed: {e}")
            logger.error(f"API request failed: {e}")
            return self._get_fallback_response(user_input)
            
        except Exception as e:
            print(f"💥 [UNEXPECTED ERROR] {e}")
            logger.error(f"Unexpected error in API client: {e}")
            return self._get_fallback_response(user_input)

    def is_healthy(self) -> bool:
        """Check if the API server is healthy by generating a short response"""
        if self.test_mode:
            return True
            
        try:
            payload = {
                "model": self.model,
                "prompt": "ping",
                "stream": False,
                "options": {"num_thread": 1, "num_ctx": 32}
            }
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return "response" in result and "error" not in result
            return False
        except:
            return False 