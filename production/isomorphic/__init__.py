"""
IsomorphicDataSet - Production Framework

A rigorous framework for proving latent space isomorphism between Large Language Models
through semantic-preserving concept variations and Procrustes alignment analysis.

Paper: "Latent Space Isomorphism in Large Language Models: Mathematical Evidence"
Submission: NeurIPS 2025

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "IsomorphicDataSet Team"
__license__ = "MIT"

from isomorphic.config import Config, ConfigManager
from isomorphic.pipeline import IsomorphicPipeline

__all__ = [
    "Config",
    "ConfigManager",
    "IsomorphicPipeline",
]
