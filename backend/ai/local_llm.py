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
    """Enhanced rule-based AI engine with better responses"""
    
    def __init__(self):
        self.responses = {
            "hello": "Hello! I'm Quantum Hive, your AI assistant. How can I help you today?",
            "how are you": "I'm functioning well, thank you for asking! How can I assist you?",
            "what can you do": "I can help with various tasks like answering questions, setting reminders, and controlling devices. What would you like to do?",
            "time": "I can't check the current time yet, but that's a feature I'm working on!",
            "weather": "I don't have access to weather information yet, but I'm learning new capabilities.",
            "help": "I'm here to help! You can ask me questions, have conversations, or request assistance with various tasks.",
            "goodbye": "Goodbye! Have a great day!",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "joke": "Why don't scientists trust atoms? Because they make up everything! 😄",
            "name": "My name is Quantum Hive, your AI assistant. Nice to meet you!",
            "who are you": "I'm Quantum Hive, an AI assistant designed to help you with various tasks and conversations.",
            "what is your name": "My name is Quantum Hive. I'm here to assist you!",
            "how does this work": "I listen to your voice, process what you say, and respond with helpful information or actions.",
            "tell me about yourself": "I'm Quantum Hive, an AI assistant built to help you. I can understand speech, process requests, and respond with useful information.",
            "capabilities": "I can understand speech, answer questions, have conversations, and I'm constantly learning new capabilities.",
            "features": "My features include speech recognition, natural language processing, and voice responses. I'm designed to be helpful and conversational."
        }
        
        # Contextual responses for better conversation flow
        self.context_responses = {
            "greeting": [
                "Hello there! How can I assist you today?",
                "Hi! I'm ready to help. What would you like to know?",
                "Greetings! I'm Quantum Hive, your AI assistant. How can I be of service?"
            ],
            "confused": [
                "I'm not quite sure I understood that. Could you rephrase your question?",
                "I'm still learning and that's a bit beyond my current capabilities. Could you try asking something else?",
                "I didn't catch that clearly. Could you say it differently?"
            ],
            "positive": [
                "That's great to hear! Is there anything specific I can help you with?",
                "Excellent! I'm glad I could be helpful. What else would you like to know?",
                "Wonderful! I'm here whenever you need assistance."
            ]
        }
    
    def cleanup(self):
        """Clean up resources (no-op for simple AI)"""
        pass
    
    def generate_response(self, prompt, system_prompt=None):
        """Generate enhanced response based on keywords and context"""
        prompt_lower = prompt.lower()
        
        # Check for exit commands first
        exit_commands = ["goodbye", "exit", "quit", "stop", "bye", "see you"]
        for cmd in exit_commands:
            if cmd in prompt_lower:
                return "Goodbye! Have a great day!"
        
        # Check for exact matches
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        # Check for greeting patterns
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(word in prompt_lower for word in greeting_words):
            import random
            return random.choice(self.context_responses["greeting"])
        
        # Check for positive sentiment
        positive_words = ["good", "great", "excellent", "awesome", "amazing", "wonderful", "fantastic"]
        if any(word in prompt_lower for word in positive_words):
            import random
            return random.choice(self.context_responses["positive"])
        
        # Check for questions about capabilities
        capability_words = ["can you", "do you", "are you able", "what can", "how can"]
        if any(phrase in prompt_lower for phrase in capability_words):
            return "I can understand speech, answer questions, have conversations, and assist with various tasks. I'm constantly learning and improving!"
        
        # Check for questions about the system
        system_words = ["how does", "how do you", "what is", "explain", "tell me about"]
        if any(phrase in prompt_lower for phrase in system_words):
            return "I'm an AI assistant that uses speech recognition to understand your voice, processes your requests, and responds with helpful information."
        
        # Check for joke requests
        if "joke" in prompt_lower or "funny" in prompt_lower:
            return self.responses["joke"]
        
        # Check for name/identity questions
        if "name" in prompt_lower or "who are you" in prompt_lower:
            return self.responses["name"]
        
        # Default response for unrecognized input
        import random
        return random.choice(self.context_responses["confused"])

# Factory function to get the best available AI engine
def get_ai_engine():
    """
    Get the best available AI engine
    
    Returns:
        EnhancedAIEngine or SimpleAIEngine: Available AI engine
    """
    try:
        from .enhanced_ai import EnhancedAIEngine
        return EnhancedAIEngine()
    except Exception as e:
        logger.warning(f"Enhanced AI engine not available: {e}")
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