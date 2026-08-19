"""Wraps Qwen 2.5-3B + your LoRA adapter as a callable parser."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from . import config


class ParserLLM:
    def __init__(self,
                 base_model: str = config.PARSER_BASE,
                 adapter_path: str | None = None):
        if adapter_path is None:
            adapter_path = config.resolve_parser_adapter()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="cuda",
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.adapter_source = adapter_path

    def _generate(self, messages, max_new_tokens: int = 10) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

    def parse(self, utterance: str, options=("A", "B")) -> tuple:
        """Returns (parsed, raw_text). parsed is a subset-list, [], "*", or None."""
        options_str = "[" + ", ".join(f'"{o}"' for o in options) + "]"
        user_msg = f"Options: {options_str}\nUser: {utterance}"
        messages = [
            {"role": "system", "content": config.PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = self._generate(messages, max_new_tokens=10)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None, raw

        valid_letters = set(options)
        if parsed == "*":
            return "*", raw
        if isinstance(parsed, list) and all(x in valid_letters for x in parsed):
            return parsed, raw
        return None, raw


def parsed_to_y(parsed, options=("A", "B")) -> tuple[int | None, str]:
    """Map parser output → oracle answer y for a 2-item query.

    Returns (y, status). y=None means skip the belief update; the step still
    counts and (i, j) are still marked used.
    """
    if parsed is None:
        return None, "malformed"
    if parsed == "*":
        return None, "uncertain"
    if parsed == []:
        return None, "reject_all"
    if not isinstance(parsed, list):
        return None, "malformed"
    if len(parsed) == 2:
        return None, "both"
    if len(parsed) == 1 and parsed[0] in options:
        return options.index(parsed[0]), "clean"
    return None, "malformed"
