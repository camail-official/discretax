"""Regression head."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.heads.base import Head, HeadConfig


@dataclass(frozen=True)
class RegressionHeadConfig(HeadConfig):
    """Configuration for the regression head.

    Attributes:
        out_features:
          Output dimensionality (prediction dimension).
    """

    def build(self, in_features: int, key: PRNGKeyArray) -> RegressionHead:
        """Build head from config.

        Args:
            in_features:
              Input dimensionality (hidden dimension).
            key:
              JAX random key for initialization.

        Returns:
            The regression head instance.
        """
        return RegressionHead(
            in_features=in_features, out_features=self.out_features, cfg=self, key=key
        )


class RegressionHead[ConfigType: RegressionHeadConfig](Head):
    """Regression head.

    This regression head takes an input of shape (timesteps, in_features)
    and outputs a regression of shape (out_features).

    Args:
        in_features:
          Input features.
        out_features:
          Output features.
        cfg:
          Configuration for the regression head.
        key:
          JAX random key for initialization.

    Attributes:
        linear:
          Linear layer.
    """

    linear: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the regression head."""
        self.linear = eqx.nn.Linear(in_features=in_features, out_features=out_features, key=key)

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the regression head.

        This forward pass applies the linear layer to the input
        and returns the mean of the output.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jnp.mean(x, axis=0)  # shape (timestep, in_features) -> (in_features)
        x = self.linear(x)  # shape (in_features) -> (out_features)
        return x, state
