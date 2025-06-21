# coding: utf-8
"""Multi-prompt parsing utilities.

This module implements a simple version of the
`MultiPromptParser` class used in the original CoXQL project.  The parser
supports generating several candidate parses from different prompt variations
and ranking these candidates according to their cosine similarity with a set of
pre-encoded parse templates.  The candidate that both passes the grammar check
and achieves the best similarity score is returned.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np
from lark import Lark
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, MaxLengthCriteria

from parsing.guided_decoding.gd_logits_processor import (
    GuidedParser,
    GuidedDecodingLogitsProcessor,
)


class MultiPromptParser:
    """Generate and rank parses from multiple prompts."""

    def __init__(
        self,
        model_name: str,
        *,
        prompt_fns: Iterable[Callable[[str], str]] | None = None,
        template_texts: Iterable[str] | None = None,
        device: str = "cpu",
        use_guided_decoding: bool = True,
        max_attempts: int = 2,
        sentence_transformer_model: str = "all-mpnet-base-v2",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.device = device
        self.use_guided_decoding = use_guided_decoding
        self.max_attempts = max_attempts
        self.prompt_fns = list(prompt_fns) if prompt_fns is not None else [lambda x: x]

        self.embedder = SentenceTransformer(sentence_transformer_model)
        if template_texts is None:
            template_texts = []
        self.template_texts = list(template_texts)
        if len(self.template_texts) > 0:
            self.template_embeddings = self.embedder.encode(self.template_texts)
        else:
            self.template_embeddings = np.empty((0, self.embedder.get_sentence_embedding_dimension()))

    @staticmethod
    def _template_check(text: str, grammar: str) -> bool:
        """Return ``True`` if ``text`` conforms to ``grammar``."""
        parse_str = text.split("parsed:")[-1].split("[e]")[0] + "[e]"
        try:
            parser = Lark(grammar, parser="lalr")
            parser.parse(parse_str)
            return True
        except Exception:
            return False

    def _generate(self, prompt: str, grammar: str) -> str:
        """Generate a parse for ``prompt`` using the underlying model."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        parser = None
        if self.use_guided_decoding:
            parser = GuidedParser(grammar, self.tokenizer, model="gpt")
            processor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])
        stopping = MaxLengthCriteria(max_length=200)
        last_decoded = ""
        for _ in range(self.max_attempts):
            if self.use_guided_decoding:
                generation = self.model.greedy_search(
                    input_ids,
                    logits_processor=processor,
                    eos_token_id=parser.eos_token,
                )
            else:
                generation = self.model.greedy_search(
                    input_ids,
                    stopping_criteria=stopping,
                )
            last_decoded = self.tokenizer.decode(generation[0])
            if self._template_check(last_decoded, grammar):
                break
        return last_decoded

    def __call__(self, prompt: str, grammar: str) -> dict:
        """Parse ``prompt`` and return the best candidate generation."""
        candidates: List[str] = []
        for fn in self.prompt_fns:
            try:
                cur_prompt = fn(prompt)
            except Exception:
                cur_prompt = prompt
            candidates.append(self._generate(cur_prompt, grammar))

        if len(self.template_embeddings) == 0:
            # No ranking possible; return the first valid generation or fallback
            for cand in candidates:
                if self._template_check(cand, grammar):
                    return {"generation": cand}
            return {"generation": candidates[0]}

        cand_embs = []
        for cand in candidates:
            parse_str = cand.split("parsed:")[-1].split("[e]")[0] + "[e]"
            emb = self.embedder.encode(parse_str)
            cand_embs.append(emb)
        cand_embs = np.stack(cand_embs)
        sims = util.cos_sim(cand_embs, self.template_embeddings)
        best_idx = sims.max(dim=1).values.argmax().item()
        return {"generation": candidates[best_idx]}


def get_mp_plus_predict_f(
    *,
    model: str,
    device: str = "cpu",
    use_guided_decoding: bool = True,
    max_attempts: int = 2,
    prompt_fns: Iterable[Callable[[str], str]] | None = None,
    template_texts: Iterable[str] | None = None,
    sentence_transformer_model: str = "all-mpnet-base-v2",
):
    """Return a prediction function performing multi-prompt MP+ parsing."""
    parser = MultiPromptParser(
        model,
        prompt_fns=prompt_fns,
        template_texts=template_texts,
        device=device,
        use_guided_decoding=use_guided_decoding,
        max_attempts=max_attempts,
        sentence_transformer_model=sentence_transformer_model,
    )

    def predict_f(prompt: str, grammar: str):
        return parser(prompt, grammar)

    return predict_f
