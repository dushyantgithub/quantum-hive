#!/usr/bin/env python3
"""
Gemma 3n Model Download Script
Downloads the Gemma 3n model in the background for Quantum Hive
Optimized for edge devices like Raspberry Pi 4
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_gemma3n():
    """Download Gemma 3n model"""
    print("🤖 Quantum Hive - Gemma 3n Model Download")
    print("=" * 50)
    print()
    
    try:
        print("📦 Installing/checking dependencies...")
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        print("✅ Dependencies available")
        
        print("\n🔐 Checking Hugging Face authentication...")
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        # Choose model size based on user preference or default to E2B
        model_name = "google/gemma-3n-E2B"  # 2B effective parameters, optimized for edge devices
        print(f"\n📥 Downloading Gemma 3n processor ({model_name})...")
        processor = AutoProcessor.from_pretrained(model_name)
        print("✅ Processor downloaded")
        
        print(f"\n📥 Downloading Gemma 3n model (this may take several minutes)...")
        print("💡 The E2B model is about 6GB but runs with ~2GB effective memory - perfect for Pi 4!")
        print("🎯 This model supports text, images, and audio input with MatFormer efficiency!")
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Better for CPU/Pi 4
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ Model downloaded successfully!")
        
        print("\n🧪 Testing model...")
        # Test with multimodal capabilities
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello! I'm Quantum Hive running on edge hardware. Can you introduce yourself?"}]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                low_memory=True
            )
        
        response_tokens = generation[0][input_len:]
        response = processor.decode(response_tokens, skip_special_tokens=True)
        print(f"✅ Test response: {response.strip()}")
        
        print("\n🎉 Gemma 3n model ready for Quantum Hive!")
        print("\n🚀 Features enabled:")
        print("  • Text generation and conversation")
        print("  • Image analysis and vision tasks")
        print("  • Audio processing and transcription")
        print("  • Optimized for Raspberry Pi 4")
        print("  • MatFormer architecture with selective parameter activation")
        
        print("\nTo use Gemma 3n in your application:")
        print('  python -c "from backend.ai.local_llm import get_ai_engine; ai = get_ai_engine(\'gemma\')"')
        
        # Clean up
        del model
        del processor
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install torch transformers>=4.53.0 accelerate huggingface_hub Pillow")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        if "401" in str(e) or "gated repo" in str(e).lower():
            print("\n🔑 Authentication Issue:")
            print("1. Go to https://huggingface.co/google/gemma-3n-E2B")
            print("2. Accept the license agreement")
            print("3. Run: huggingface-cli login")
            print("4. Enter your Hugging Face token")
        elif "transformers" in str(e).lower():
            print("\n📦 Version Issue:")
            print("Gemma 3n requires transformers>=4.53.0")
            print("Run: pip install --upgrade transformers>=4.53.0")
        return False
    
    return True

if __name__ == "__main__":
    success = download_gemma3n()
    if success:
        print("\n✅ Download completed successfully!")
        print("🎯 Gemma 3n is now ready for edge deployment!")
    else:
        print("\n❌ Download failed. Please check the instructions above.")
        sys.exit(1) 