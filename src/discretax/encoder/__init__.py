"""This module contains the encoders implemented in Discretax."""

from discretax.encoder.base import AbstractEncoder
from discretax.encoder.embedding import EmbeddingEncoder
from discretax.encoder.linear import LinearEncoder

__all__ = [
    "AbstractEncoder",
    "LinearEncoder",
    "EmbeddingEncoder",
]
