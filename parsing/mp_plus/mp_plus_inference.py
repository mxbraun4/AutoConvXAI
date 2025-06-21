"""MP+ inference with template checking."""
import gin
import torch
from lark import Lark
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria

# Import LlamaTokenizer explicitly for Mistral models
try:
    from transformers import LlamaTokenizer
except ImportError:
    LlamaTokenizer = None

from parsing.guided_decoding.gd_logits_processor import (
    GuidedParser,
    GuidedDecodingLogitsProcessor,
)


@gin.configurable
def get_mp_plus_predict_f(model: str, device: str = "cpu", use_guided_decoding: bool = True, max_attempts: int = 2):
    """Gets a prediction function implementing MP+ with template checking.

    Args:
        model: Path or name of the base language model.
        device: Device to load the model on.
        use_guided_decoding: Whether to use guided decoding.
        max_attempts: Number of attempts to satisfy the template check.
    """
    print(f"Loading MP+ model from: {model}")
    print(f"Target device: {device}")
    
    # Try to load tokenizer with explicit trust_remote_code for newer models
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"Failed to load tokenizer with trust_remote_code: {e}")
        # Fallback for older transformers versions
        try:
            if LlamaTokenizer is not None:
                tokenizer = LlamaTokenizer.from_pretrained(model)
                print("✓ Tokenizer loaded with LlamaTokenizer fallback")
            else:
                raise ValueError(f"Cannot load tokenizer for model {model}. Consider updating transformers library.")
        except Exception as e2:
            raise ValueError(f"Failed to load tokenizer: {e2}")
    
    print("Loading model...")
    try:
        # Optimized model loading for better memory management
        base_model = AutoModelForCausalLM.from_pretrained(
            model, 
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to reduce memory
            low_cpu_mem_usage=True,     # Enable low memory loading
            device_map="auto" if device != "cpu" else None  # Auto device mapping for GPU
        )
        print("✓ Model loaded with optimized settings")
    except Exception as e:
        print(f"Failed to load model with optimized settings: {e}")
        print("Trying fallback loading...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ Model loaded with fallback settings")
        except Exception as e2:
            print(f"Failed with fallback: {e2}")
            raise ValueError(f"Could not load model {model}: {e2}")
    
    # Move to device if not using device_map
    if device == "cpu" or "device_map" not in locals():
        base_model = base_model.to(device)
        print(f"✓ Model moved to {device}")
    
    base_model.config.pad_token_id = base_model.config.eos_token_id
    print("✓ MP+ model ready for inference")

    def _template_check(text: str, grammar: str) -> bool:
        """Returns True if the generated text can be parsed by the grammar."""
        parse_str = text.split("parsed:")[-1].split("[e]")[0] + "[e]"
        try:
            parser = Lark(grammar, parser="lalr")
            parser.parse(parse_str)
            return True
        except Exception:
            return False

    def predict_f(prompt: str, grammar: str):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        parser = None
        if use_guided_decoding:
            parser = GuidedParser(grammar, tokenizer, model="gpt")
            processor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])
        stopping = MaxLengthCriteria(max_length=200)
        last_decoded = ""
        for _ in range(max_attempts):
            if use_guided_decoding:
                generation = base_model.greedy_search(
                    input_ids,
                    logits_processor=processor,
                    eos_token_id=parser.eos_token,
                )
            else:
                generation = base_model.greedy_search(
                    input_ids,
                    stopping_criteria=stopping,
                )
            last_decoded = tokenizer.decode(generation[0])
            if _template_check(last_decoded, grammar):
                break
        return {"generation": last_decoded}

    return predict_f
