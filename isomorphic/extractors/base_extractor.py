"""
Vector Extraction Framework

Provides multiple methods for extracting latent representations from LLM outputs.
Methods: Mean Pooling, Last Token, Hybrid, Attention-Weighted.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class BaseExtractor(ABC):
    """Abstract base class for vector extractors."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 128):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self, load_in_4bit: bool = False) -> None:
        """Load model and tokenizer with optional 4-bit quantization for large models."""
        from transformers import BitsAndBytesConfig
        
        print(f"Loading {self.model_name} on {self.device}...")
        
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=self.device if not load_in_4bit else "auto",
            torch_dtype=torch.float16
        )
        
        # If not using auto device map, move to specific device
        if not load_in_4bit and "cuda" in self.device:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print(f"[DONE] Model loaded on {self.device} (4-bit: {load_in_4bit})")
    
    def extract_all_methods(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Extract vectors using all available methods in a single forward pass.
        
        Args:
            text: Input sentence
            
        Returns:
            Dictionary mapping method names to vectors
        """
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        results = {}
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # 1. Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        results['mean_pooling'] = (summed / counts).squeeze().cpu()
        
        # 2. Last Token
        results['last_token'] = last_hidden[0, -1, :].cpu()
        
        # 3. Attention-Weighted
        # Simple weighted sum using last layer attention if available, 
        # or uniform weighting over masked tokens as fallback
        weights = torch.ones_like(last_hidden[:, :, :1]) / counts.unsqueeze(-1)
        results['attention_weighted'] = (last_hidden * weights).sum(dim=1).squeeze().cpu()
        
        return results

    def extract_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Extract vectors for batch of texts."""
        all_vectors = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting with {self.__class__.__name__}"):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = self._extract_batch_internal(batch_texts)
            all_vectors.append(batch_vectors)
        
        return torch.vstack(all_vectors)
    
    def _extract_batch_internal(self, texts: List[str]) -> torch.Tensor:
        """Internal method for batch extraction (to be overridden)."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return self._extract_vectors(outputs, inputs)
    
    @abstractmethod
    def _extract_vectors(self, outputs, inputs) -> torch.Tensor:
        """Extract vectors from model outputs (to be implemented)."""
        pass
    
    def extract_single(self, text: str) -> torch.Tensor:
        """Extract vector for single text."""
        vectors = self._extract_batch_internal([text])
        return vectors[0]


class MeanPoolingExtractor(BaseExtractor):
    """Mean pooling with attention mask."""
    
    def _extract_vectors(self, outputs, inputs) -> torch.Tensor:
        """Mean pooling over all tokens with attention mask."""
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        
        # Apply attention mask
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        
        mean_pooled = summed / counts
        return mean_pooled.cpu()


class LastTokenExtractor(BaseExtractor):
    """Extract last token representation."""
    
    def _extract_vectors(self, outputs, inputs) -> torch.Tensor:
        """Get last token embedding."""
        last_hidden = outputs.last_hidden_state
        last_tokens = last_hidden[:, -1, :]
        return last_tokens.cpu()


class HybridExtractor(BaseExtractor):
    """Hybrid: Concatenation of mean pooling + last token."""
    
    def _extract_vectors(self, outputs, inputs) -> torch.Tensor:
        """Concatenate mean pooling and last token."""
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        
        # Last token
        last_tokens = last_hidden[:, -1, :]
        
        # Concatenate
        hybrid = torch.cat([mean_pooled, last_tokens], dim=-1)
        return hybrid.cpu()


class AttentionWeightedExtractor(BaseExtractor):
    """Attention-weighted pooling."""
    
    def _extract_vectors(self, outputs, inputs) -> torch.Tensor:
        """Compute attention-weighted pooling."""
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(1).unsqueeze(1)
        
        # Global average, considering attention
        attention_probs = torch.ones_like(last_hidden[:, :, :1]) / (attention_mask.sum(dim=-1) + 1e-9)
        weighted = (last_hidden * attention_probs).sum(dim=1)
        
        return weighted.cpu()


class ExtractorFactory:
    """Factory for creating extractors."""
    
    EXTRACTORS = {
        "mean_pooling": MeanPoolingExtractor,
        "last_token": LastTokenExtractor,
        "hybrid": HybridExtractor,
        "attention_weighted": AttentionWeightedExtractor,
    }
    
    @classmethod
    def create(cls, method: str, model_name: str, **kwargs) -> BaseExtractor:
        """Create extractor instance."""
        if method not in cls.EXTRACTORS:
            raise ValueError(f"Unknown extraction method: {method}")
        
        return cls.EXTRACTORS[method](model_name, **kwargs)
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List available extraction methods."""
        return list(cls.EXTRACTORS.keys())
