"""
IsomorphicDataSet - Production Framework

A rigorous framework for proving latent space isomorphism between Large Language Models
through semantic-preserving concept variations and Procrustes alignment analysis.

Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "IsomorphicDataSet Team"
__license__ = "MIT"

__all__ = [
    "Config",
    "ConfigManager",
    "IsomorphicPipeline",
]


def __getattr__(name: str):
    if name == "Config":
        from isomorphic.config import Config

        return Config
    if name == "ConfigManager":
        from isomorphic.config import ConfigManager

        return ConfigManager
    if name == "IsomorphicPipeline":
        from isomorphic.pipeline import IsomorphicPipeline

        return IsomorphicPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
