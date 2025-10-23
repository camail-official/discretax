"""Linear encoder."""

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.encoder.base import Encoder, EncoderConfig


@dataclass(frozen=True)
class LinearEncoderConfig(EncoderConfig):
    """Configuration for the linear encoder."""

    in_features: int
    use_bias: bool = False

    def build(self, out_features: int, key: PRNGKeyArray) -> "LinearEncoder":
        """Build encoder from config."""
        return LinearEncoder(out_features=out_features, cfg=self, key=key)


class LinearEncoder[ConfigType: LinearEncoderConfig](Encoder):
    """Linear encoder.

    This encoder takes an input of shape (timesteps, in_features)
    and outputs a hidden representation of shape (timesteps, hidden_dim).

    Attributes:
        linear:
          MLP instance with multiple hidden layers and a last linear layer.
    """

    linear: eqx.nn.Linear

    def __init__(self, out_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the linear encoder.

        Args:
            out_features:
              Output dimensionality.
            cfg:
              Configuration for the linear encoder.
            key:
              JAX random key for initialization.
        """
        self.linear = eqx.nn.Linear(
            in_features=cfg.in_features,
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
