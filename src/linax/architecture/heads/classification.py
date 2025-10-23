"""Classification head."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.heads.base import Head, HeadConfig


@dataclass(frozen=True)
class ClassificationHeadConfig(HeadConfig):
    """Configuration for the classification head."""


class ClassificationHead[ConfigType: ClassificationHeadConfig](Head):
    """Classification head.

    This classification head takes an input of shape (timesteps, in_features)
    and outputs a logits of shape (out_features).

    Args:
        in_features:
          Input features.
        cfg:
          Configuration for the classification head.
        key:
          JAX random key for initialization.

    Attributes:
        linear:
          Linear layer.

    """

    linear: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
        self.linear = eqx.nn.Linear(in_features, cfg.out_features, key=key)

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the classification head.

        This forward pass applies the linear layer to the input
        and returns the logits of the output.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.vmap(self.linear)(x)
        x = jnp.mean(x, axis=0)
        x = jax.nn.log_softmax(x, axis=-1)
        return x, state
