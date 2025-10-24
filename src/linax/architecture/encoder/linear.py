"""Linear encoder."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.encoder.base import Encoder, EncoderConfig


@dataclass(frozen=True)
class LinearEncoderConfig(EncoderConfig):
    """Configuration for the linear encoder.

    Attributes:
        in_features:
          Input dimensionality (number of input features).
        out_features:
          Output dimensionality (hidden dimension).
        use_bias:
          Whether to use bias in the linear layer.
    """

    in_features: int
    use_bias: bool = False

    def build(self, key: PRNGKeyArray) -> LinearEncoder:
        """Build encoder from config.

        Args:
            key:
              JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        return LinearEncoder(
            in_features=self.in_features, out_features=self.out_features, cfg=self, key=key
        )


class LinearEncoder[ConfigType: LinearEncoderConfig](Encoder):
    """Linear encoder.

    This encoder takes an input of shape (timesteps, in_features)
    and outputs a hidden representation of shape (timesteps, hidden_dim).

    Attributes:
        linear:
          MLP instance with multiple hidden layers and a last linear layer.

    Args:
        in_features:
          Input dimensionality.
        out_features:
          Output dimensionality.
        cfg:
          Configuration for the linear encoder.
        key:
          JAX random key for initialization.
    """

    linear: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the linear encoder."""
        self.linear = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            key=key,
            use_bias=cfg.use_bias,
        )

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the linear encoder.

        This forward pass applies the linear layer to the input.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.vmap(self.linear)(x)
        return x, state
