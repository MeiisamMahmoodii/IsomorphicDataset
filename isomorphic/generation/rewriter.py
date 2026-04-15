"""
Model Rewriter - High-Fidelity Variation Generation

Uses diverse LLMs (Llama-3.2, Gemma-4, Qwen-3.5) to rewrite sentences
under strict length and banned-word constraints.
"""

import torch
from typing import List, Dict, Any, Optional

from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint
from isomorphic.generation.constraint_utils import (
    WRITER_MAX_ATTEMPTS,
    rewrite_passes_constraints,
)
from isomorphic.utils.device_utils import get_model_input_device, normalize_device_map


class ModelRewriter:
    """
    Handles systematic rewriting of sentences across multiple models and constraints.
    """

    MODELS = [
        "huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated",
        "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
        "huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated",
        "huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-E4B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-31B-it-abliterated",
        "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
    ]
    # Constant instruction prepended to every rewrite request.
    COMMON_REWRITE_INSTRUCTION = (
        "You are a strict paraphrasing system.\n"
        "Return exactly ONE rewritten sentence only.\n"
        "Do not add explanations, notes, labels, or lists.\n"
        "Keep the same semantic meaning as the source sentence.\n"
    )
    # Globally banned generic filler words, always applied in addition
    # to per-sentence key-word bans.
    GLOBAL_GENERIC_BANNED_WORDS = [
        "thing",
        "things",
        "stuff",
        "very",
        "really",
        "just",
        "maybe",
    ]

    def __init__(
        self,
        current_model_id: str,
        model=None,
        tokenizer=None,
        device: str = "cuda:0",
        load_in_4bit: bool = False,
        device_map: Optional[Any] = None,
        verbose_attempts: bool = True,
    ):
        self.model_id = current_model_id
        # `device` is used only for inputs when the model is single-device.
        # For sharded models, we derive the correct input device from parameters.
        self.device = device
        self.device_map = device_map
        self.verbose_attempts = verbose_attempts

        if model is None:
            self._load_local_model(load_in_4bit)
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.input_device = get_model_input_device(self.model)

    def _load_local_model(self, load_in_4bit: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except AttributeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, extra_special_tokens={}
            )

        # If an explicit device_map was provided, it takes precedence.
        # Otherwise we fall back to `device` for single-GPU placement.
        dm = self.device_map if self.device_map is not None else self.device
        device_map = normalize_device_map(dm)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.input_device = get_model_input_device(self.model)
        print(f"[DONE] Rewriter loaded: {self.model_id} (device_map={dm}, input_device={self.input_device})")

    @staticmethod
    def build_rewrite_prompt(text: str, banned_words: List[str], constraint: LengthConstraint) -> str:
        """Exact instruction string persisted in Phase A and used in Phase B."""
        merged = []
        seen = set()
        for w in list(banned_words) + list(ModelRewriter.GLOBAL_GENERIC_BANNED_WORDS):
            n = str(w).strip().lower()
            if not n or n in seen:
                continue
            merged.append(n)
            seen.add(n)
        banned_str = ", ".join(merged)
        return (
            f"{ModelRewriter.COMMON_REWRITE_INSTRUCTION}"
            f"Rewrite the following sentence using between {constraint.min_words} "
            f"and {constraint.max_words} words.\n"
            f"DO NOT use any of these words: {banned_str}.\n"
            f"Sentence: {text}\n"
            f"Rewritten:"
        )

    def rewrite_with_prompt(self, prompt: str, constraint: LengthConstraint) -> str:
        """Generate continuation from a full prompt (used when rewrite_specs are fixed)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        max_tokens = int(constraint.max_words * 1.5) + 10

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        variation = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        return variation.strip()

    def rewrite(self, text: str, banned_words: List[str], constraint: LengthConstraint) -> str:
        """Rewrite using the standard template (backward compatible)."""
        prompt = self.build_rewrite_prompt(text, banned_words, constraint)
        return self.rewrite_with_prompt(prompt, constraint)

    def process_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Generate variations with up to WRITER_MAX_ATTEMPTS per length bin."""
        effective_banned_words = []
        seen_banned = set()
        for w in list(entry.forbidden_words) + list(self.GLOBAL_GENERIC_BANNED_WORDS):
            n = str(w).strip().lower()
            if not n or n in seen_banned:
                continue
            effective_banned_words.append(n)
            seen_banned.add(n)

        for constraint in entry.length_constraints:
            desc = f"{constraint.min_words}-{constraint.max_words}"
            key = f"{self.model_id}_{desc}"

            prompt: Optional[str] = None
            if entry.rewrite_specs and desc in entry.rewrite_specs:
                prompt = entry.rewrite_specs[desc]
            else:
                prompt = self.build_rewrite_prompt(
                    entry.seed_text, effective_banned_words, constraint
                )

            log: Dict[str, Any] = {"attempts": [], "success": False}
            variation: Optional[str] = None

            for attempt in range(1, WRITER_MAX_ATTEMPTS + 1):
                out = self.rewrite_with_prompt(prompt, constraint)
                ok, reason = rewrite_passes_constraints(
                    out, effective_banned_words, constraint.min_words, constraint.max_words
                )
                log["attempts"].append({"attempt": attempt, "output_preview": out[:500], "ok": ok, "reason": reason})
                if self.verbose_attempts:
                    print("\n--- rewrite attempt ---")
                    print(f"model: {self.model_id}")
                    print(f"attempt: {attempt}/{WRITER_MAX_ATTEMPTS}")
                    print(f"original: {entry.seed_text}")
                    print(f"banned_key_words: {entry.forbidden_words}")
                    print(f"banned_global_words: {self.GLOBAL_GENERIC_BANNED_WORDS}")
                    print(f"banned_effective: {effective_banned_words}")
                    print(f"rewritten: {out}")
                    print(f"status: {'PASS' if ok else 'FAIL'} ({reason})")
                if ok:
                    variation = out
                    log["success"] = True
                    break

            if variation is None:
                variation = log["attempts"][-1]["output_preview"] if log["attempts"] else ""
                log["success"] = False

            entry.variations[key] = variation
            entry.rewrite_logs[f"{self.model_id}::{key}"] = log

        return entry
