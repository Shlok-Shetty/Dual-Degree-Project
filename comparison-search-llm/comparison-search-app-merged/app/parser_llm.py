"""Wraps Qwen 2.5-3B + your LoRA adapter as a callable parser.

Backends (see devices.py): 4-bit on CUDA, fp16 on Apple MPS, bf16 on CPU.
On MPS/CPU the LoRA adapter is merged into the base weights; on 4-bit CUDA it
stays separate, because merging into 4-bit weights changes outputs.

On MPS/CPU the system prompt's KV cache is computed once and reused (prefix
cache), so only the short user turn is processed per parse. Measured on the
210-example test set + 28 extra cases (fp16, merged): identical outputs with
and without the cache, median parse 524 -> 429 ms. On 4-bit CUDA the cache is
off by default: it saved only ~4% and changed 1 of 238 outputs.

Note: the adapter was trained on the 4-bit base (QLoRA). On the fp16 path it
gives different answers on 7 of those 238 cases (4-option test set through
parse(): 204/210 vs 207/210 on 4-bit CUDA), mostly ordinal negations like
"the third one is off". parse() caps output at 10 tokens, which is enough for
the app's 2-option answers but truncates a 4-letter list.
"""
import copy
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel

from . import config
from .answers import parsed_to_y  # noqa: F401  (re-exported for older imports)
from .devices import pick_device, parser_dtype


class ParserLLM:
    def __init__(self,
                 base_model: str = config.PARSER_BASE,
                 adapter_path: str | None = None,
                 use_prefix_cache: bool | None = None):
        if adapter_path is None:
            adapter_path = config.resolve_parser_adapter()

        self.device = pick_device()
        if use_prefix_cache is None:
            use_prefix_cache = self.device != "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb_config, device_map="cuda",
            )
            model = PeftModel.from_pretrained(base, adapter_path)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=parser_dtype(self.device),
            ).to(self.device)
            model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

        self.model = model.eval()
        self.adapter_source = adapter_path

        self._prefix_ids = None
        self._prefix_cache = None
        if use_prefix_cache:
            self._build_prefix_cache()

    def _messages(self, user_msg: str):
        return [
            {"role": "system", "content": config.PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _build_prefix_cache(self):
        """Encode everything up to the user's text once and keep its KV cache."""
        marker = "\u0000PARSER_USER_TURN\u0000"
        text = self.tokenizer.apply_chat_template(
            self._messages(marker), add_generation_prompt=True, tokenize=False,
        )
        prefix_text = text[: text.index(marker)]
        ids = self.tokenizer(prefix_text, add_special_tokens=False,
                             return_tensors="pt")["input_ids"]
        # Leave the last prefix token out of the cache so the per-call input is
        # never empty and token boundaries at the join are re-checked each call.
        ids = ids[:, :-1].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=DynamicCache(),
                             use_cache=True)
        self._prefix_ids = ids
        self._prefix_cache = out.past_key_values

    def _generate(self, messages, max_new_tokens: int = 10) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        n = 0 if self._prefix_ids is None else self._prefix_ids.shape[1]
        use_cache = (
            n > 0
            and input_ids.shape[1] > n
            and torch.equal(input_ids[:, :n], self._prefix_ids)
        )
        with torch.no_grad():
            if use_cache:
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    past_key_values=copy.deepcopy(self._prefix_cache),
                    **gen_kwargs,
                )
            else:
                out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

    def parse(self, utterance: str, options=("A", "B")) -> tuple:
        """Returns (parsed, raw_text). parsed is a subset-list, [], "*", or None."""
        options_str = "[" + ", ".join(f'"{o}"' for o in options) + "]"
        user_msg = f"Options: {options_str}\nUser: {utterance}"
        raw = self._generate(self._messages(user_msg), max_new_tokens=10)
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
