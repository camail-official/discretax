"""SwiGLU (Swish Gated Linear Unit) layer.

SwiGLU is a variant of the Gated Linear Unit (GLU) that uses the Swish activation
function instead of sigmoid.

References:
    Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202. https://arxiv.org/abs/2002.05202
    Aziz et al. Paper Summary: https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
"""

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray


class SwiGLU(eqx.Module):
    """Swish Gated Linear Unit (SwiGLU) layer.

    Adapted from https://huggingface.co/blog/sachithgunasekara/nanojaxgpt .

    The architecture consists of three linear projections:
    - gate_proj: Projects input to intermediate dimension
    - up_proj: Projects input to intermediate dimension
    - down_proj: Projects intermediate dimension back to hidden dimension
    The computation is: down_proj(swish(gate_proj(x)) * up_proj(x))

    Args:
        hidden_dim:
          Dimensionality of the input and output features.
        hidden_ratio:
          Ratio to scale hidden dimension for intermediate size calculation.
          If None, defaults to 4.
        intermediate_dim:
          Dimensionality of the intermediate projection. If None, calculated as
          `int(hidden_dim * hidden_ratio * 2/3)` rounded to nearest multiple of 256.
        key:
          JAX random key for weight initialization.
    """

    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(
        self,
        hidden_dim: int,
        hidden_ratio: int | float | None,
        intermediate_dim: int | None,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)

        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_dim is None:
            intermediate_dim = int(hidden_dim * hidden_ratio * 2 / 3)
            intermediate_dim = 256 * ((intermediate_dim + 256 - 1) // 256)

        self.gate_proj = eqx.nn.Linear(hidden_dim, intermediate_dim, use_bias=False, key=k1)
        self.up_proj = eqx.nn.Linear(hidden_dim, intermediate_dim, use_bias=False, key=k2)
        self.down_proj = eqx.nn.Linear(intermediate_dim, hidden_dim, use_bias=False, key=k3)

    def __call__(self, x: Array) -> Array:
        """Forward pass of the SwiGLU layer.

        Args:
            x:
              Input tensor.

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
