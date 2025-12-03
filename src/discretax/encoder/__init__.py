"""This module contains the encoders implemented in Discretax."""

from discretax.encoder.base import Encoder, EncoderConfig
from discretax.encoder.embedding import (
    EmbeddingEncoder,
    EmbeddingEncoderConfig,
)
from discretax.encoder.linear import LinearEncoder, LinearEncoderConfig

__all__ = [
    "EncoderConfig",
    "Encoder",
    "LinearEncoder",
    "LinearEncoderConfig",
    "EmbeddingEncoder",
    "EmbeddingEncoderConfig",
]
