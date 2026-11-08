"""Qwen 2.5-1.5B, prompted (no fine-tune) to produce natural utterances styled
from a rotating pool. Used only in LLM/simulated-user mode.

Includes an optional parser-based validation gate: if a parser is passed in,
verbalize() will regenerate up to MAX_RETRIES times if the parser can't recover
the picked letter from the utterance. Falls back to a bare letter if all
retries flip. This exists because the 1.5B verbalizer occasionally flips
options on 'ordinal' style (e.g. says "the second one" when the user picked A)
- see the flip-source decomposition in the closed-loop eval notebook.

Backends (see devices.py): fp16 on CUDA and Apple MPS, bf16 on CPU.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .devices import pick_device, verbalizer_dtype


MAX_RETRIES = 3


class VerbalizerLLM:
    def __init__(self, model_id: str = config.VERBALIZER_MODEL):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="cuda",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=verbalizer_dtype(self.device),
            ).to(self.device)
        self.model.eval()
        self.styles = config.VERBALIZER_STYLES
        # (parsed, raw) from the gate's parse of the returned utterance, or
        # None if no parse was done (no parser given, or bare-letter fallback).
        self.last_parse = None

    def _generate_one(self, picked_letter: str, style_instruction: str, seed: int) -> str:
        other = "B" if picked_letter == "A" else "A"
        messages = [
            {"role": "system", "content":
                f"You are simulating a user who has just picked option {picked_letter} out of two options A and B. "
                f"They did NOT pick {other}. "
                f"Reminder: A is the first / left option. B is the second / right option. "
                f"Style: {style_instruction} "
                f"Output only the user's response, nothing else."},
            {"role": "user", "content": "Response:"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=15,
                do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return text.strip().strip('"').strip("'")

    def verbalize(self, picked_letter: str, rng: np.random.Generator,
                    parser=None) -> tuple[str, str]:
        """Simulate a user who just picked `picked_letter` (A or B).
        Returns (utterance, style_name).

        If `parser` is passed, the utterance is validated: the parser must
        recover exactly [picked_letter] from it. Up to MAX_RETRIES regenerations
        with different seeds. If all retries flip, falls back to a bare letter
        (deterministic). The style_name returned is "letter (fallback)" in
        that case so downstream can log the fallback happened.

        After a call, `self.last_parse` holds the gate's (parsed, raw) for the
        returned utterance so callers do not need to parse it again.
        """
        self.last_parse = None
        style_name, style_instruction = self.styles[rng.integers(0, len(self.styles))]

        if parser is None:
            seed = int(rng.integers(0, 1_000_000))
            return self._generate_one(picked_letter, style_instruction, seed), style_name

        for _ in range(MAX_RETRIES):
            seed = int(rng.integers(0, 1_000_000))
            utterance = self._generate_one(picked_letter, style_instruction, seed)
            parsed, raw = parser.parse(utterance, options=("A", "B"))
            if isinstance(parsed, list) and parsed == [picked_letter]:
                self.last_parse = (parsed, raw)
                return utterance, style_name

        return picked_letter, "letter (fallback)"
