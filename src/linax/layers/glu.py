"""Gated Linear Unit (GLU) layer.

See: copied from LinOSS implementation.
"""

import equinox as eqx
import jax
import jax.random as jr


class GLU(eqx.Module):
    """Gated Linear Unit (GLU) layer."""

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        """Initialize the GLU layer."""
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        """Forward pass of the GLU layer."""
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))
