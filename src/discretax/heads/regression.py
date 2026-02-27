"""Regression head."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from discretax.heads.base import AbstractHead


class RegressionHead(AbstractHead):
    """Regression head.

    This regression head takes an input of shape (timesteps, in_features)
    and outputs a regression of shape (out_features).

    Attributes:
        linear: Linear layer.
        reduce: Whether to reduce the time dimension by averaging.
    """

    linear: eqx.nn.Linear
    reduce: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        reduce: bool = True,
        **kwargs,
    ):
        """Initialize the regression head.

        Args:
            in_features: input features.
            out_features: output features (prediction dimension).
            key: JAX random key for initialization.
            reduce: whether to reduce the time dimension by averaging.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.linear = eqx.nn.Linear(in_features=in_features, out_features=out_features, key=key)
        self.reduce = reduce

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the regression head.

        This forward pass applies the linear layer to the input
        and returns the mean of the output.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for stochastic operations (unused).

        Returns:
            Tuple containing the output tensor and updated state. If reduce is True,
            the output tensor is of shape (out_features). If reduce is False,
            the output tensor is of shape (timesteps, out_features).
        """
        # reduce over the time dimension if reduce is True
        if self.reduce:
            x = jnp.mean(x, axis=0)  # shape (timestep, in_features) -> (in_features)
        x = self.linear(x)  # shape ((timesteps), in_features)) -> ((timesteps),out_features)
        return x, state
