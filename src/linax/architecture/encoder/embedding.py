"""Embedding encoder."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.encoder.base import Encoder, EncoderConfig


@dataclass(frozen=True)
class EmbeddingEncoderConfig(EncoderConfig):
    """Configuration for the embedding encoder.

    Attributes:
        in_features:
          Number of classes (vocabulary size). Inherited but semantically represents num_classes.
        out_features:
          Output dimensionality (embedding dimension).
    """

    num_classes: int

    def build(self, key: PRNGKeyArray) -> EmbeddingEncoder:
        """Build encoder from config.

        Args:
            key:
              JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        # in_features represents num_classes for embedding encoder
        return EmbeddingEncoder(
            num_classes=self.num_classes, out_features=self.out_features, cfg=self, key=key
        )


class EmbeddingEncoder[ConfigType: EmbeddingEncoderConfig](Encoder):
    """Embedding encoder.

    This encoder takes an input of shape (timesteps,)
    and outputs a hidden representation of shape (timesteps, out_features).

    Attributes:
        embedding:
          Embedding layer.

    Args:
        num_classes:
          Number of classes (vocabulary size).
        out_features:
          Output dimensionality (embedding dimension).
        cfg:
          Configuration for the embedding encoder.
        key:
          JAX random key for initialization.
    """

    def __init__(self, num_classes: int, out_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the embedding encoder."""
        self.embedding = eqx.nn.Embedding(
            num_classes=num_classes, embedding_size=out_features, key=key
        )

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the embedding encoder.

        This forward pass applies the embedding layer to the input.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.vmap(self.embedding)(x)  # vmap over the timestep dimension
        return x, state
