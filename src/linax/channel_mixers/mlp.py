"""MLP channel mixer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.channel_mixers.base import ChannelMixer, ChannelMixerConfig

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


@dataclass(frozen=True)
class MLPChannelMixerConfig(ChannelMixerConfig):
    """Configuration for the MLP channel mixer.

    Attributes:
        non_linearity: Name of the activation function to apply after the linear layer.
        use_bias: Whether to include a bias term in the linear layer.
    """

    non_linearity: activation = "gelu"
    use_bias: bool = False

    def build(
        self, in_features: int, out_features: int | None, key: PRNGKeyArray
    ) -> MLPChannelMixer:
        """Build MLPChannelMixer from config.

        Args:
            in_features: Input dimensionality.
            out_features: Optional output dimensionality. If None, defaults to in_features.
            key: JAX random key for initialization.

        Returns:
            The MLPChannelMixer instance.
        """
        return MLPChannelMixer(
            in_features=in_features, cfg=self, key=key, out_features=out_features
        )


class MLPChannelMixer[ConfigType: MLPChannelMixerConfig](ChannelMixer):
    """MLP channel mixer.

    This channel mixer applies a multi-layer perceptron (MLP) to the input tensor.

    Args:
        in_features: The input dimensionality.
        cfg: Configuration for the MLP channel mixer.
        key: JAX random key for initialization.
        out_features: Optional output dimensionality. If None, defaults to in_features.

    Attributes:
        linear: Linear layer applied to the input.
        non_linearity: The non-linearity function used after the linear layer.
    """

    linear: eqx.nn.Linear
    non_linearity: activation

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        *,
        out_features: int | None = None,
    ):
        """Initialize the MLP channel mixer."""
        out_dim = out_features if out_features is not None else in_features
        self.linear = eqx.nn.Linear(in_features, out_dim, use_bias=cfg.use_bias, key=key)

        self.non_linearity = cfg.non_linearity

    def __call__(self, x: Array) -> Array:
        """Forward pass of the MLP channel mixer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return _get_activation(self.non_linearity)(self.linear(x))
