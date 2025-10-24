"""MLP channel mixer."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray


def non_linearity_factory(
    non_linearity: Literal["relu", "gelu", "swish", "silu", "tanh"],
) -> Callable[[Array], Array]:
    """Factory function for non-linearities.

    Args:
        non_linearity: The non-linearity to use.

    Returns:
        A function that applies the non-linearity to the input.

    Raises:
        ValueError: If the non-linearity is invalid.
    """
    if non_linearity == "relu":
        return jax.nn.relu
    elif non_linearity == "gelu":
        return jax.nn.gelu
    elif non_linearity == "swish":
        return jax.nn.swish
    elif non_linearity == "silu":
        return jax.nn.silu
    elif non_linearity == "tanh":
        return jax.nn.tanh
    else:
        raise ValueError(f"Invalid non-linearity: {non_linearity}")


class MLPChannelMixer(eqx.Module):
    """MLP channel mixer.

    Args:
        input_dim: Dimensionality of the input features.
        output_dim: Dimensionality of the output features.
        key: JAX random key for initialization.

    Attributes:
        w1: First linear layer.
        w2: Second linear layer.
    This channel mixer applies a multi-layer perceptron (MLP) to the input tensor.
    """

    linear: eqx.nn.Linear
    non_linearity: Literal["relu", "gelu", "swish", "silu", "tanh"]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        non_linearity: Literal["relu", "gelu", "swish", "silu", "tanh"],
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the MLP channel mixer."""
        self.linear = eqx.nn.Linear(input_dim, output_dim, use_bias=use_bias, key=key)

        self.non_linearity = non_linearity

    def __call__(self, x: Array) -> Array:
        """Forward pass of the MLP channel mixer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return non_linearity_factory(self.non_linearity)(self.linear(x))
