"""Qwen 2.5-1.5B, prompted (no fine-tune) to produce natural utterances styled
from a rotating pool. Used only in LLM/simulated-user mode."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


class VerbalizerLLM:
    def __init__(self, model_id: str = config.VERBALIZER_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
        )
        self.model.eval()
        self.styles = config.VERBALIZER_STYLES

    def verbalize(self, picked_letter: str, rng: np.random.Generator) -> tuple[str, str]:
        """Simulate a user who just picked `picked_letter` (A or B).
        Returns (utterance, style_name)."""
        style_name, style_instruction = self.styles[rng.integers(0, len(self.styles))]
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
        ).to("cuda")
        torch.manual_seed(int(rng.integers(0, 1_000_000)))
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=15,
                do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return text.strip().strip('"').strip("'"), style_name
