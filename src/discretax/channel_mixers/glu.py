"""Gated Linear Unit (GLU) layer.

Adapted from LinOSS: https://github.com/tk-rusch/linoss
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.channel_mixers.base import AbstractChannelMixer


class GLU(AbstractChannelMixer):
    """Gated Linear Unit (GLU) layer.

    Attributes:
        w1: First linear layer.
        w2: Second linear layer.

    Source:
        https://arxiv.org/pdf/2002.05202
    """

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        out_features: int | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize the GLU layer.

        Args:
            in_features: dimensionality of the input features.
            key: JAX random key for initialization.
            out_features: optional dimensionality of the output features (defaults to in_features).
            use_bias: whether to include a bias term in the linear layers.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        w1_key, w2_key = jr.split(key, 2)

        out_features = out_features if out_features is not None else in_features
        self.w1 = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=w1_key)
        self.w2 = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=w2_key)

    def __call__(self, x: Array) -> Array:
        """Forward pass of the GLU layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying gated linear transformation.
        """
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))

    def __repr__(self) -> str:
        """Return a string representation of the GLU layer.

        Returns:
            Compact summary showing dimensions.
        """
        in_dim = self.w1.in_features
        out_dim = self.w1.out_features
        return f"GLU({in_dim}→{out_dim})"
