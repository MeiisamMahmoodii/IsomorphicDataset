"""
Model Rewriter - High-Fidelity Variation Generation

Uses diverse LLMs (Llama-3.2, Gemma-4, Qwen-3.5) to rewrite sentences
under strict length and banned-word constraints.
"""

import torch
from typing import List, Dict
from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint

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
        "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
    ]

    def __init__(self, current_model_id: str, model=None, tokenizer=None, device: str = "cuda:0", load_in_4bit: bool = False):
        self.model_id = current_model_id
        self.device = device
        
        if model is None:
            self._load_local_model(load_in_4bit)
        else:
            self.model = model
            self.tokenizer = tokenizer

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
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=self.device if not load_in_4bit else "auto",
            torch_dtype=torch.float16
        )
        print(f"[DONE] Rewriter loaded: {self.model_id} on {self.device}")

    def rewrite(self, text: str, banned_words: List[str], constraint: LengthConstraint) -> str:
        """
        Rewrite a sentence while adhering to length and word constraints.
        """
        banned_str = ", ".join(banned_words)
        prompt = f"Rewrite the following sentence using between {constraint.min_words} " \
                 f"and {constraint.max_words} words. DO NOT use any of these words: {banned_str}.\n" \
                 f"Sentence: {text}\n" \
                 f"Rewritten:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Calculate max tokens based on max words (approx 1.5 tokens per word)
        max_tokens = int(constraint.max_words * 1.5) + 10
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        variation = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return variation.strip()

    def process_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Generate variations for a dataset entry across all length constraints."""
        for constraint in entry.length_constraints:
            desc = f"{constraint.min_words}-{constraint.max_words}"
            key = f"{self.model_id}_{desc}"
            
            variation = self.rewrite(entry.seed_text, entry.forbidden_words, constraint)
            entry.variations[key] = variation
            
        return entry
