"""Gated Linear Unit (GLU) layer.

Adapted from LinOSS: https://github.com/tk-rusch/linoss
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.channel_mixers.base import ChannelMixer, ChannelMixerConfig


@dataclass(frozen=True)
class GLUConfig(ChannelMixerConfig):
    """Configuration for the GLU channel mixer.

    Attributes:
        use_bias: Whether to include a bias term in the linear layers.
    """

    use_bias: bool = True

    def build(self, in_features: int, out_features: int | None, key: PRNGKeyArray) -> GLU:
        """Build GLU from config.

        Args:
            in_features: Input dimensionality.
            out_features: Optional output dimensionality. If None, defaults to in_features.
            key: JAX random key for initialization.

        Returns:
            The GLU instance.
        """
        return GLU(in_features=in_features, cfg=self, key=key, out_features=out_features)


class GLU[ConfigType: GLUConfig](ChannelMixer):
    """Gated Linear Unit (GLU) layer.

    Args:
        in_features: Dimensionality of the input features.
        out_features: Optional dimensionality of the output features (defaults to in_features).
        key: JAX random key for initialization.

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
        cfg: ConfigType,
        key: PRNGKeyArray,
        *,
        out_features: int | None = None,
    ):
        """Initialize the GLU layer."""
        w1_key, w2_key = jr.split(key, 2)

        out_features = out_features if out_features is not None else in_features
        self.w1 = eqx.nn.Linear(in_features, out_features, use_bias=cfg.use_bias, key=w1_key)
        self.w2 = eqx.nn.Linear(in_features, out_features, use_bias=cfg.use_bias, key=w2_key)

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
