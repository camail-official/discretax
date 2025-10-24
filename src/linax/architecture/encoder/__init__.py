"""This module contains the encoders implemented in Linax."""

from linax.architecture.encoder.base import Encoder, EncoderConfig
from linax.architecture.encoder.embedding import (
    EmbeddingEncoder,
    EmbeddingEncoderConfig,
)
from linax.architecture.encoder.linear import LinearEncoder, LinearEncoderConfig

__all__ = [
    "EncoderConfig",
    "Encoder",
    "LinearEncoder",
    "LinearEncoderConfig",
    "EmbeddingEncoder",
    "EmbeddingEncoderConfig",
]
