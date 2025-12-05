"""This module contains the encoders implemented in Discretax."""

from discretax.encoder.base import Encoder
from discretax.encoder.embedding import EmbeddingEncoder
from discretax.encoder.linear import LinearEncoder

__all__ = [
    "Encoder",
    "LinearEncoder",
    "EmbeddingEncoder",
]
