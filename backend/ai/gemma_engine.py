"""
Gemma 3n AI Engine
High-quality multimodal local LLM using Google's Gemma 3n model via transformers
Optimized for edge devices like Raspberry Pi 4
"""
import logging
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import warnings
import time
from PIL import Image
import io
import base64

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class Gemma3nAIEngine:
    """Gemma 3n AI engine using transformers library with multimodal support"""
    
    def __init__(self, model_name="google/gemma-3n-E2B", use_cpu_only=True, enable_vision=True, enable_audio=True):
        """
        Initialize Gemma 3n AI engine
        
        Args:
            model_name (str): Hugging Face model name (E2B or E4B)
            use_cpu_only (bool): Force CPU-only mode (recommended for Raspberry Pi)
            enable_vision (bool): Enable vision processing capabilities
            enable_audio (bool): Enable audio processing capabilities
        """
        self.model_name = model_name
        self.use_cpu_only = use_cpu_only
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio
        self.conversation_history = []
        self.max_history_length = 6  # Keep last 6 exchanges for context
        
        # Determine the best device
        self.device = self._get_best_device()
        
        # Initialize model and processor
        self._load_model()
        
        logger.info(f"Gemma 3n AI engine initialized with model: {model_name} on {self.device}")
        logger.info(f"Vision enabled: {enable_vision}, Audio enabled: {enable_audio}")
    
    def _get_best_device(self):
        """Determine the best available device"""
        if self.use_cpu_only:
            return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _load_model(self):
        """Load Gemma 3n model and processor"""
        try:
            logger.info(f"Loading Gemma 3n model: {self.model_name}")
            
            # Load processor (handles text, images, and audio)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure device mapping for parameter efficiency
            device_map_config = {
                "": self.device  # Load all parameters to the same device
            }
            
            # Load model with optimizations for edge devices
            if self.device == "cpu":
                # CPU mode - optimized for Raspberry Pi
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Better compatibility on CPU
                    device_map=device_map_config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    # Conditional parameter loading - skip unused modalities to save memory
                    load_vision_params=self.enable_vision,
                    load_audio_params=self.enable_audio,
                )
            else:
                # GPU mode - optimized for performance
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map_config,
                    trust_remote_code=True,
                    load_vision_params=self.enable_vision,
                    load_audio_params=self.enable_audio,
                )
            
            # Enable evaluation mode for inference
            self.model.eval()
            
            logger.info(f"Gemma 3n model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n model: {e}")
            # Fallback: try without conditional parameter loading
            try:
                logger.info("Retrying without conditional parameter loading...")
                if self.device == "cpu":
                    self.model = Gemma3ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=device_map_config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                else:
                    self.model = Gemma3ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map_config,
                        trust_remote_code=True
                    )
                self.model.eval()
                logger.info("Gemma 3n model loaded successfully (standard loading)")
            except Exception as e2:
                logger.error(f"Failed to load Gemma 3n model with fallback: {e2}")
                raise
    
    def _prepare_messages(self, user_input, system_prompt=None, image=None, audio=None):
        """
        Prepare messages for Gemma 3n chat template
        
        Args:
            user_input (str): User's text input
            system_prompt (str): Optional system prompt
            image (PIL.Image or str): Optional image input
            audio (bytes or str): Optional audio input
            
        Returns:
            list: Formatted messages for the model
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"System instructions: {system_prompt}\n\nPlease follow these instructions in all your responses."}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "I understand the instructions and will follow them in my responses."}]
            })
        
        # Add conversation history for context
        for exchange in self.conversation_history[-self.max_history_length:]:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": exchange["user"]}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": exchange["assistant"]}]
            })
        
        # Prepare current user message content
        current_content = []
        
        # Add image if provided
        if image and self.enable_vision:
            if isinstance(image, str):
                # Assume it's a base64 encoded image or file path
                if image.startswith('data:image'):
                    # Base64 encoded image
                    current_content.append({"type": "image", "image": image})
                else:
                    # File path
                    try:
                        img = Image.open(image)
                        current_content.append({"type": "image", "image": img})
                    except Exception as e:
                        logger.warning(f"Could not load image from path {image}: {e}")
            elif isinstance(image, Image.Image):
                current_content.append({"type": "image", "image": image})
        
        # Add audio if provided
        if audio and self.enable_audio:
            current_content.append({"type": "audio", "audio": audio})
        
        # Add text input
        current_content.append({"type": "text", "text": user_input})
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": current_content
        })
        
        return messages
    
    def generate_response(self, user_input, system_prompt=None, max_length=200, temperature=0.7, image=None, audio=None):
        """
        Generate AI response using Gemma 3n
        
        Args:
            user_input (str): User's input message
            system_prompt (str): Optional system prompt
            max_length (int): Maximum response length
            temperature (float): Sampling temperature (0.0 to 1.0)
            image (PIL.Image or str): Optional image input
            audio (bytes or str): Optional audio input
            
        Returns:
            str: AI response
        """
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = self._prepare_messages(user_input, system_prompt, image, audio)
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get input length for response extraction
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response with optimized settings for edge devices
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    top_k=50,
                    # Memory optimization for edge devices
                    use_cache=True,
                    low_memory=True if self.device == "cpu" else False
                )
            
            # Extract and decode only the new tokens (response)
            response_tokens = generation[0][input_len:]
            response = self.processor.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            # Add to conversation history
            if response and len(response.strip()) > 0:
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                
                # Keep history manageable
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _clean_response(self, response):
        """
        Clean and format the AI response
        
        Args:
            response (str): Raw response from model
            
        Returns:
            str: Cleaned response
        """
        # Remove common artifacts
        response = response.strip()
        
        # Remove any remaining chat template artifacts
        cleanup_patterns = [
            "<start_of_turn>model\n",
            "<end_of_turn>",
            "<start_of_turn>user",
            "<start_of_turn>assistant",
            "System instructions:",
            "I understand the instructions"
        ]
        
        for pattern in cleanup_patterns:
            response = response.replace(pattern, "")
        
        # Remove leading/trailing whitespace again
        response = response.strip()
        
        # Ensure reasonable length (avoid overly long responses)
        if len(response) > 400:
            # Find the last complete sentence within 400 characters
            truncated = response[:400]
            sentence_endings = ['.', '?', '!']
            last_sentence_end = -1
            
            for ending in sentence_endings:
                pos = truncated.rfind(ending)
                if pos > last_sentence_end:
                    last_sentence_end = pos
            
            if last_sentence_end > 50:  # Make sure we have a reasonable amount of text
                response = response[:last_sentence_end + 1]
            else:
                response = truncated + "..."
        
        return response
    
    def process_image(self, image_path, prompt="Describe this image"):
        """
        Process an image with Gemma 3n vision capabilities
        
        Args:
            image_path (str): Path to image file
            prompt (str): Text prompt for image analysis
            
        Returns:
            str: AI response about the image
        """
        if not self.enable_vision:
            return "Vision capabilities are not enabled for this model instance."
        
        try:
            # Load image
            image = Image.open(image_path)
            return self.generate_response(prompt, image=image)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "I couldn't process the image. Please check the file path and try again."
    
    def process_audio(self, audio_data, prompt="Transcribe this audio"):
        """
        Process audio with Gemma 3n audio capabilities
        
        Args:
            audio_data (bytes): Audio data
            prompt (str): Text prompt for audio analysis
            
        Returns:
            str: AI response about the audio
        """
        if not self.enable_audio:
            return "Audio capabilities are not enabled for this model instance."
        
        try:
            return self.generate_response(prompt, audio=audio_data)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "I couldn't process the audio. Please try again."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "vision_enabled": self.enable_vision,
            "audio_enabled": self.enable_audio,
            "history_length": len(self.conversation_history),
            "max_history": self.max_history_length,
            "effective_params": "2B" if "E2B" in self.model_name else "4B",
            "architecture": "Gemma 3n (MatFormer)"
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        logger.info("Gemma 3n AI engine cleaned up")

# Backward compatibility alias
GemmaAIEngine = Gemma3nAIEngine

if __name__ == "__main__":
    # Test the Gemma 3n AI engine
    logging.basicConfig(level=logging.INFO)
    print("Testing Gemma 3n AI Engine...")
    
    try:
        # Initialize engine
        ai_engine = Gemma3nAIEngine(use_cpu_only=True)
        
        # Test text queries
        test_queries = [
            "Hello! What can you help me with?",
            "What is the capital of France?",
            "Can you explain quantum computing in simple terms?",
            "Tell me a joke about AI",
            "Thank you for the help!"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            response = ai_engine.generate_response(query)
            print(f"Gemma 3n: {response}")
            
        # Test with system prompt
        print(f"\n--- Testing with system prompt ---")
        system_prompt = "You are a helpful AI assistant named Quantum Hive. Keep responses concise and friendly. You are optimized for edge devices."
        response = ai_engine.generate_response("Who are you?", system_prompt=system_prompt)
        print(f"User: Who are you?")
        print(f"Gemma 3n: {response}")
        
        # Display model info
        print(f"\n--- Model Information ---")
        info = ai_engine.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error testing Gemma 3n engine: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ai_engine' in locals():
            ai_engine.cleanup() 