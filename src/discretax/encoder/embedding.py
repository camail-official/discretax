"""Embedding encoder."""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from discretax.encoder.base import AbstractEncoder


class EmbeddingEncoder(AbstractEncoder):
    """Embedding encoder.

    This encoder takes an input of shape (timesteps,)
    and outputs a hidden representation of shape (timesteps, out_features).

    Attributes:
        embedding: Embedding layer.
    """

    embedding: eqx.nn.Embedding

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        out_features: int,
        num_classes: int,
        **kwargs,
    ):
        """Initialize the embedding encoder.

        Args:
            key: JAX random key for initialization.
            out_features: output dimensionality (embedding dimension).
            num_classes: number of classes (vocabulary size).
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.embedding = eqx.nn.Embedding(
            num_embeddings=num_classes, embedding_size=out_features, key=key
        )

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the embedding encoder.

        This forward pass applies the embedding layer to the input.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for stochastic operations (unused).

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.vmap(self.embedding)(x)  # vmap over the timestep dimension
        return x, state
