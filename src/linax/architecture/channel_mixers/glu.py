"""Gated Linear Unit (GLU) layer.

Adapted from LinOSS: https://github.com/tk-rusch/linoss
"""

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class GLU(eqx.Module):
    """Gated Linear Unit (GLU) layer.

    Attributes:
        w1:
          First linear layer.
        w2:
          Second linear layer.

    Args:
        input_dim:
          Dimensionality of the input features.
        output_dim:
          Dimensionality of the output features.
        key:
          JAX random key for initialization.

    Source:
        https://arxiv.org/pdf/2002.05202
    """

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim: int, output_dim: int, key: PRNGKeyArray):
        """Initialize the GLU layer."""
        w1_key, w2_key = jr.split(key, 2)

        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x: Array) -> Array:
        """Forward pass of the GLU layer.

        Args:
            x:
              Input tensor.

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
