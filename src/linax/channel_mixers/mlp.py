"""MLP channel mixer."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

# the available activations
activation = Literal["relu", "gelu", "swish", "silu", "tanh"]

# activation registry to map string names to activation functions
ACTIVATION_REGISTRY = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "silu": jax.nn.silu,
    "tanh": jax.nn.tanh,
}


def _get_activation(
    non_linearity: activation,
) -> Callable[[Array], Array]:
    """Get the activation function from the registry.

    This function is used to retrieve the activation function from the registry.

    Args:
        non_linearity: name of the activation function.

    Returns:
        The activation function.

    Raises:
        KeyError: If the activation function is invalid.
    """
    try:
        return ACTIVATION_REGISTRY[non_linearity]
    except KeyError:
        raise KeyError(
            f"Invalid activation: {non_linearity}."
            f" Valid activations are: {list(ACTIVATION_REGISTRY.keys())}."
        )


class MLPChannelMixer(eqx.Module):
    """MLP channel mixer.

    This channel mixer applies a multi-layer perceptron (MLP) to the input tensor.

    Args:
        input_dim: Dimensionality of the input features.
        output_dim: Dimensionality of the output features.
        key: JAX random key for initialization.

    Attributes:
        linear: Linear layer applied to the input.
        non_linearity: The non-linearity function used after the linear layer.
    """

    linear: eqx.nn.Linear
    non_linearity: activation

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        non_linearity: activation,
        use_bias: bool = False,
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
        return _get_activation(self.non_linearity)(self.linear(x))
