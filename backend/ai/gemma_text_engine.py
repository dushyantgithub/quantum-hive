"""
TinyLlama Text-Only AI Engine
Simplified engine for TinyLlama-1.1B-Chat-v1.0 (fast chat-tuned LLM)
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import time

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class TinyLlamaTextAIEngine:
    """TinyLlama 1.1B chat-tuned AI engine using transformers library"""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_cpu_only=True, generation_config=None):
        self.model_name = model_name
        self.use_cpu_only = use_cpu_only
        self.conversation_history = []
        self.max_history_length = 4
        self.generation_config = {
            "max_new_tokens": 40,
            "temperature": 0.5,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
        }
        if generation_config:
            self.generation_config.update(generation_config)
        self.device = self._get_best_device()
        self._load_model()
        logger.info(f"TinyLlama text AI engine initialized with model: {model_name} on {self.device}")
    
    def _get_best_device(self):
        if self.use_cpu_only:
            return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        try:
            logger.info(f"Loading TinyLlama model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            self.model.eval()
            logger.info(f"TinyLlama model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load TinyLlama model: {e}")
            raise
    
    def _format_conversation(self, user_input, system_prompt=None):
        conversation = ""
        if system_prompt:
            conversation += f"<|system|> {system_prompt}\n"
        for exchange in self.conversation_history[-self.max_history_length:]:
            conversation += f"<|user|> {exchange['user']}\n"
            conversation += f"<|assistant|> {exchange['assistant']}\n"
        conversation += f"<|user|> {user_input}\n<|assistant|>"
        return conversation
    
    def generate_response(self, user_input, system_prompt=None, history=None, **generation_kwargs):
        try:
            start_time = time.time()
            # If history is provided, prepend it to the conversation
            prompt = ""
            if system_prompt:
                prompt += f"<|system|> {system_prompt}\n"
            if history:
                for user, ai in history:
                    prompt += f"<|user|> {user}\n<|assistant|> {ai}\n"
            prompt += f"<|user|> {user_input}\n<|assistant|>"
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)
            gen_config = self.generation_config.copy()
            gen_config.update(generation_kwargs)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            response = self._clean_response(response)
            if response and len(response.strip()) > 0:
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _clean_response(self, response):
        response = response.strip()
        cleanup_patterns = [
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "\n\n"
        ]
        for pattern in cleanup_patterns:
            if response.startswith(pattern):
                response = response[len(pattern):].strip()
        if len(response) > 200:
            truncated = response[:200]
            sentence_endings = ['.', '?', '!']
            last_sentence_end = -1
            for ending in sentence_endings:
                pos = truncated.rfind(ending)
                if pos > last_sentence_end:
                    last_sentence_end = pos
            if last_sentence_end > 20:
                response = response[:last_sentence_end + 1]
            else:
                response = truncated + "..."
        return response
    
    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "history_length": len(self.conversation_history),
            "max_history": self.max_history_length,
            "architecture": "TinyLlama 1.1B Chat",
            "generation_config": self.generation_config
        }
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        logger.info("TinyLlama text AI engine cleaned up")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from backend.utils.gemma_config import get_tinyllama_config
    logging.basicConfig(level=logging.INFO)
    print("Testing TinyLlama 1.1B Chat AI Engine...")
    try:
        config = get_tinyllama_config("development")
        ai_engine = TinyLlamaTextAIEngine(
            model_name=config["model_name"],
            use_cpu_only=config["use_cpu_only"],
            generation_config=config["generation_config"]
        )
        test_queries = [
            "Hello! How are you?",
            "What is the capital of France?",
            "Tell me a short joke",
            "Thank you!"
        ]
        for query in test_queries:
            print(f"\nUser: {query}")
            response = ai_engine.generate_response(query)
            print(f"TinyLlama: {response}")
        print(f"\n--- Model Information ---")
        info = ai_engine.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error testing TinyLlama engine: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ai_engine' in locals():
            ai_engine.cleanup() 