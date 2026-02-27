"""Linear encoder."""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from discretax.encoder.base import AbstractEncoder


class LinearEncoder(AbstractEncoder):
    """Linear encoder.

    This encoder takes an input of shape (timesteps, in_features)
    and outputs a hidden representation of shape (timesteps, hidden_dim).

    Attributes:
        linear: Linear layer.
    """

    linear: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        out_features: int,
        use_bias: bool = False,
        **kwargs,
    ):
        """Initialize the linear encoder.

        Args:
            in_features: input dimensionality (number of input features).
            key: JAX random key for initialization.
            out_features: output dimensionality (hidden dimension).
            use_bias: whether to use bias in the linear layer.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.linear = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            key=key,
            use_bias=use_bias,
        )

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the linear encoder.

        This forward pass applies the linear layer to the input.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for stochastic operations (unused).

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.vmap(self.linear)(x)
        return x, state
