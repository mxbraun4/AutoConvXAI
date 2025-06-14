"""MP+ inference with template checking."""
import gin
from lark import Lark
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria

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
    tokenizer = AutoTokenizer.from_pretrained(model)
    base_model = AutoModelForCausalLM.from_pretrained(model)
    base_model = base_model.to(device)
    base_model.config.pad_token_id = base_model.config.eos_token_id

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
