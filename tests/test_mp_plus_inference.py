"""Tests for MP+ inference generation"""
from os.path import dirname, abspath
import sys

import os
from lark import Lark
import pytest

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from parsing.mp_plus.mp_plus_inference import get_mp_plus_predict_f  # noqa: E402


@pytest.mark.parametrize("prompt", ["parsed:"])
def test_mp_plus_generation(prompt):
    grammar = """
    ?start: animal " [e]"
    animal: " cat" | " dog"
    %ignore " "
    """

    predict_f = get_mp_plus_predict_f(
        model="hf-internal-testing/tiny-random-gpt2",
        use_guided_decoding=True,
    )

    result = predict_f(prompt=prompt, grammar=grammar)
    generated = result["generation"]

    parse_str = generated.split("parsed:")[-1].split("[e]")[0] + " [e]"
    parser = Lark(grammar, parser="lalr")
    parser.parse(parse_str)

    cleaned = parse_str.strip()
    assert cleaned.startswith("cat") or cleaned.startswith("dog")
