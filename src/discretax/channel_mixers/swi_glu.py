"""SwiGLU (Swish Gated Linear Unit) layer.

SwiGLU is a variant of the Gated Linear Unit (GLU) that uses the Swish activation
function instead of sigmoid.

References:
    Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202. https://arxiv.org/abs/2002.05202
    Aziz et al. Paper Summary: https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
"""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from discretax.channel_mixers.base import AbstractChannelMixer


class SwiGLU(AbstractChannelMixer):
    """Swish Gated Linear Unit (SwiGLU) layer.

    Adapted from https://huggingface.co/blog/sachithgunasekara/nanojaxgpt .

    The architecture consists of three linear projections:
    - gate_proj: Projects input to intermediate dimension
    - up_proj: Projects input to intermediate dimension
    - down_proj: Projects intermediate dimension back to hidden dimension
    The computation is: down_proj(swish(gate_proj(x)) * up_proj(x))

    Attributes:
        gate_proj: Linear layer for the gate projection.
        up_proj: Linear layer for the up projection.
        down_proj: Linear layer for the down projection.
    """

    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        out_features: int | None = None,
        intermediate_dim: int | None = None,
        use_bias: bool = False,
        hidden_ratio: int | float = 4,
        **kwargs,
    ) -> None:
        """Initialize the SwiGLU layer.

        Args:
            in_features: the input dimensionality.
            key: JAX random key for initialization.
            out_features: optional output dimensionality (unused, kept for compatibility).
            hidden_ratio: FFN expansion ratio used to compute the intermediate dimension as
                `in_features * hidden_ratio * 2/3`, rounded up to a multiple of 256. Ignored when
                `intermediate_dim` is set explicitly.
            intermediate_dim: optional explicit intermediate size. When set, `hidden_ratio`
                is ignored.
            use_bias: whether to include a bias term in the linear layers.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        k1, k2, k3 = jax.random.split(key, 3)

        if intermediate_dim is None:
            intermediate_dim = int(in_features * hidden_ratio * 2 / 3)
            intermediate_dim = 256 * ((intermediate_dim + 256 - 1) // 256)

        self.gate_proj = eqx.nn.Linear(in_features, intermediate_dim, use_bias=use_bias, key=k1)
        self.up_proj = eqx.nn.Linear(in_features, intermediate_dim, use_bias=use_bias, key=k2)
        self.down_proj = eqx.nn.Linear(intermediate_dim, in_features, use_bias=use_bias, key=k3)

    def __call__(self, x: Array) -> Array:
        """Forward pass of the SwiGLU layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor of after applying the SwiGLU transformation.
        """
        gate, y = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(jax.nn.swish(gate) * y)

    def __repr__(self) -> str:
        """Return a string representation of the SwiGLU layer.

        Returns:
            Compact summary showing dimensions.
        """
        hidden_dim = self.gate_proj.in_features
        intermediate_dim = self.gate_proj.out_features
        return f"SwiGLU({hidden_dim}→{intermediate_dim}→{hidden_dim})"
