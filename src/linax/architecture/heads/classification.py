"""Classification head."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.heads.base import Head, HeadConfig


@dataclass
class ClassificationHeadConfig(HeadConfig):
    """Configuration for the classification head."""

    name: str = "classification_head"
    in_features: int = 64
    out_features: int = 10


class ClassificationHead[ConfigType: ClassificationHeadConfig](Head):
    """Classification head.

    This classification head takes an input of shape (timesteps, in_features)
    and outputs a logits of shape (out_features).

    Attributes:
        linear:
          Linear layer.

    """

    linear: eqx.nn.Linear

    def __init__(
        self,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
        self.linear = eqx.nn.Linear(cfg.in_features, cfg.out_features, key=key)

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
