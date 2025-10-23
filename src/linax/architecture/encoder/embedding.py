"""Embedding encoder."""

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.encoder.base import Encoder, EncoderConfig


@dataclass
class EmbeddingEncoderConfig(EncoderConfig):
    """Configuration for the embedding encoder."""

    # TODO: decide where it makes sense to define defaults and where it does not.
    num_classes: int = 10
    name: str = "embedding_encoder"
    hidden_dim: int = 32


class EmbeddingEncoder[ConfigType: EmbeddingEncoderConfig](Encoder):
    """Embedding encoder.

    This encoder takes an input of shape (timesteps, in_features)
    and outputs a hidden representation of shape (timesteps, hidden_dim).

    Attributes:
        embedding:
          Embedding layer.
    """

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the embedding encoder.

        Args:
            cfg:
              Configuration for the embedding encoder.
            key:
              JAX random key for initialization.
        """
        self.embedding = eqx.nn.Embedding(
            num_classes=cfg.num_classes, embedding_size=cfg.hidden_dim, key=key
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
        x = jax.vmap(self.embedding)(x)
        return x, state
