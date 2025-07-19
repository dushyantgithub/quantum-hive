"""
Local LLM Engine using llama.cpp
"""
import logging
import json
from pathlib import Path
from ..utils.config import LOCAL_LLM_SETTINGS, AI_SETTINGS

logger = logging.getLogger(__name__)

class LocalLLMEngine:
    """Local LLM engine using llama.cpp"""
    
    def __init__(self, model_path=None):
        """Initialize local LLM engine"""
        self.model_path = model_path or LOCAL_LLM_SETTINGS["model_path"]
        self.context_length = LOCAL_LLM_SETTINGS["context_length"]
        self.threads = LOCAL_LLM_SETTINGS["threads"]
        self.temperature = AI_SETTINGS["temperature"]
        self.max_tokens = AI_SETTINGS["max_tokens"]
        
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the local LLM"""
        try:
            from llama_cpp import Llama
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"Model not found: {self.model_path}")
                logger.info("You can download models from Hugging Face or use a smaller model")
                return
            
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_threads=self.threads,
                verbose=False
            )
            logger.info(f"Local LLM initialized with model: {self.model_path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            raise RuntimeError("llama-cpp-python not available")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            raise
    
    def generate_response(self, prompt, system_prompt=None):
        """
        Generate AI response using local LLM
        
        Args:
            prompt (str): User input prompt
            system_prompt (str, optional): System prompt for context
            
        Returns:
            str: Generated response
        """
        if not self._llm:
            return "Local LLM not available. Please check model installation."
        
        try:
            # Prepare the full prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            # Generate response
            response = self._llm(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            logger.info(f"Local LLM response: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def chat(self, messages, max_history=10):
        """
        Chat with context history
        
        Args:
            messages (list): List of message dictionaries
            max_history (int): Maximum number of messages to keep in context
            
        Returns:
            str: Generated response
        """
        if not self._llm:
            return "Local LLM not available. Please check model installation."
        
        try:
            # Limit history to prevent context overflow
            recent_messages = messages[-max_history:] if len(messages) > max_history else messages
            
            # Build conversation prompt
            conversation = ""
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    conversation += f"User: {content}\n"
                elif role == 'assistant':
                    conversation += f"Assistant: {content}\n"
            
            conversation += "Assistant:"
            
            # Generate response
            response = self._llm(
                conversation,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            logger.info(f"Local LLM chat response: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I'm sorry, I encountered an error in our conversation."
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self._llm:
            return {"status": "not_loaded", "error": "Model not available"}
        
        try:
            return {
                "status": "loaded",
                "model_path": self.model_path,
                "context_length": self.context_length,
                "threads": self.threads,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        if self._llm:
            del self._llm
            self._llm = None
        logger.info("Local LLM engine cleaned up")

# Simple fallback AI for when local LLM is not available
class SimpleAIEngine:
    """Simple rule-based AI engine as fallback"""
    
    def __init__(self):
        self.responses = {
            "hello": "Hello! I'm Quantum Hive, your AI assistant. How can I help you today?",
            "how are you": "I'm functioning well, thank you for asking! How can I assist you?",
            "what can you do": "I can help with various tasks like answering questions, setting reminders, and controlling devices. What would you like to do?",
            "time": "I can't check the current time yet, but that's a feature I'm working on!",
            "weather": "I don't have access to weather information yet, but I'm learning new capabilities.",
            "help": "I'm here to help! You can ask me questions, have conversations, or request assistance with various tasks.",
            "goodbye": "Goodbye! Have a great day!",
            "thanks": "You're welcome! Is there anything else I can help you with?"
        }
    
    def generate_response(self, prompt, system_prompt=None):
        """Generate simple response based on keywords"""
        prompt_lower = prompt.lower()
        
        # Check for exact matches first
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        # Check for partial matches
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return self.responses["hello"]
        elif "how are you" in prompt_lower:
            return self.responses["how are you"]
        elif "what can you do" in prompt_lower or "capabilities" in prompt_lower:
            return self.responses["what can you do"]
        elif "time" in prompt_lower:
            return self.responses["time"]
        elif "weather" in prompt_lower:
            return self.responses["weather"]
        elif "help" in prompt_lower:
            return self.responses["help"]
        elif "bye" in prompt_lower or "goodbye" in prompt_lower:
            return self.responses["goodbye"]
        elif "thank" in prompt_lower:
            return self.responses["thanks"]
        else:
            return "I understand you said: '" + prompt + "'. I'm still learning and don't have a specific response for that yet. Could you try asking something else?"

# Factory function to get the best available AI engine
def get_ai_engine():
    """
    Get the best available AI engine
    
    Returns:
        LocalLLMEngine or SimpleAIEngine: Available AI engine
    """
    try:
        return LocalLLMEngine()
    except Exception as e:
        logger.warning(f"Local LLM not available: {e}")
        logger.info("Falling back to simple AI engine")
        return SimpleAIEngine()

if __name__ == "__main__":
    # Test the AI engine
    logging.basicConfig(level=logging.INFO)
    print("Testing AI Engine...")
    
    # Try local LLM first, fallback to simple AI
    ai_engine = get_ai_engine()
    
    test_prompts = [
        "Hello, how are you?",
        "What can you do?",
        "Tell me a joke",
        "What's the weather like?"
    ]
    
    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        response = ai_engine.generate_response(prompt)
        print(f"AI: {response}")
    
    ai_engine.cleanup() 