"""
Model Rewriter - High-Fidelity Variation Generation

Uses diverse LLMs (Llama-3.2, Gemma-4, Qwen-3.5) to rewrite sentences
under strict length and banned-word constraints.
"""

import re
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
        "You are a strict paraphrasing engine.\n"
        "Follow ALL rules exactly.\n"
        "RULE 1: Output exactly one sentence only.\n"
        "RULE 2: Keep the same meaning as the source sentence.\n"
        "RULE 3: Do NOT include explanation, note, justification, disclaimer, labels, bullets, or quotes.\n"
        "RULE 4: Do NOT repeat the prompt text.\n"
        "RULE 5: Output plain sentence text only.\n"
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
    OUTPUT_CLEAN_PREFIXES = (
        "note:",
        "explanation:",
        "justification:",
        "rewritten:",
        "rewritten sentence:",
        "original:",
        "response:",
    )
    OUTPUT_META_PATTERNS = (
        "this is one way i can rewrite",
        "while adhering to",
        "according to your",
        "i can rewrite that sentence",
        "i rewrote the sentence",
        "the rewritten sentence is",
        "here is the rewritten sentence",
        "i can rephrase",
        "human:",
        "assistant:",
        "system:",
        "user:",
        "word version",
        "recipe for",
        "i need a recipe",
        "rephrased sentence",
        "this rephrased sentence",
        "uses fewer than",
        "maintaining the original meaning",
        "</rewrite>",
        "<rewrite",
        "one sentence here",
        "now produce only",
        "human resources",
        "job satisfaction",
        "employee retention",
        "analyze the request",
        "role: strict paraphrasing engine",
        "</think>",
        "<think>",
    )

    @staticmethod
    def _ensure_set_submodule_compat() -> None:
        """
        Compatibility shim for environments where model classes don't expose
        `set_submodule`, but newer transformers loaders expect it.
        """
        try:
            from transformers.modeling_utils import PreTrainedModel  # type: ignore
        except Exception:
            return

        if hasattr(PreTrainedModel, "set_submodule"):
            return

        def _set_submodule(self, target: str, module):  # type: ignore
            parts = target.split(".")
            parent = self
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], module)
            return self

        setattr(PreTrainedModel, "set_submodule", _set_submodule)

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
        self._ensure_set_submodule_compat()

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Some models expose custom tokenizer/model classes via remote code.
        # Fall back to slow tokenizer for backends not available in the current env.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=False,
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
            attn_implementation="eager",
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
            "Rewritten sentence:"
        )

    def rewrite_with_prompt(self, prompt: str, constraint: LengthConstraint) -> str:
        """Generate continuation from a full prompt (used when rewrite_specs are fixed)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        max_tokens = int(constraint.max_words * 1.5) + 10

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            except RuntimeError as e:
                # Some checkpoints produce NaN/Inf probabilities in sampling mode.
                # Fall back to deterministic decoding to avoid hard crash.
                if "probability tensor contains either `inf`, `nan` or element < 0" not in str(e):
                    raise
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

        variation = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        return self._sanitize_output(variation.strip(), constraint)

    @classmethod
    def _sanitize_output(cls, text: str, constraint: LengthConstraint) -> str:
        """
        Remove assistant-style meta text and keep one concise sentence.
        """
        if not text:
            return ""

        # Strip accidental XML / placeholder echoes from the model.
        text = re.sub(r"</?rewrite[^>]*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bone sentence here\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"</?think[^>]*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\*\*[^*]+\*\*", " ", text)

        # Drop lines that look like explanations/labels.
        cleaned_lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            low = s.lower()
            if any(low.startswith(p) for p in cls.OUTPUT_CLEAN_PREFIXES):
                continue
            if any(p in low for p in cls.OUTPUT_META_PATTERNS):
                continue
            cleaned_lines.append(s)
        if not cleaned_lines:
            return ""

        merged = " ".join(cleaned_lines)
        # Candidate sentence chunks; prefer one within target length.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", merged) if p.strip()]
        candidates = []
        for p in parts or [merged.strip()]:
            p = re.sub(r"^(Rewritten|Output|Answer)\s*:\s*", "", p, flags=re.IGNORECASE).strip()
            low = p.lower()
            if any(meta in low for meta in cls.OUTPUT_META_PATTERNS):
                continue
            if any(low.startswith(pref) for pref in cls.OUTPUT_CLEAN_PREFIXES):
                continue
            candidates.append(p)

        if not candidates:
            return ""

        # Choose first candidate in range, else shortest candidate.
        chosen = None
        for c in candidates:
            n = len(c.split())
            if constraint.min_words <= n <= constraint.max_words:
                chosen = c
                break
        if chosen is None:
            chosen = min(candidates, key=lambda c: len(c.split()))

        # Hard cap to max words for validation friendliness.
        words = chosen.split()
        if len(words) > constraint.max_words:
            chosen = " ".join(words[: constraint.max_words]).strip()
        return chosen

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
                # Do not store a failed last attempt as a "valid" rewrite (it misleads logs/exports).
                variation = ""
                log["success"] = False
                log["failure_reason"] = (
                    log["attempts"][-1].get("reason", "no_pass")
                    if log["attempts"]
                    else "no_attempts"
                )

            entry.variations[key] = variation
            entry.rewrite_logs[f"{self.model_id}::{key}"] = log

        return entry
