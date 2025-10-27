"""SwiGLU (Swish Gated Linear Unit) layer.

SwiGLU is a variant of the Gated Linear Unit (GLU) that uses the Swish activation
function instead of sigmoid.

References:
    Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202. https://arxiv.org/abs/2002.05202
    Aziz et al. Paper Summary: https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.channel_mixers.base import ChannelMixer, ChannelMixerConfig


@dataclass(frozen=True)
class SwiGLUConfig(ChannelMixerConfig):
    """Configuration for the SwiGLU channel mixer.

    Attributes:
        hidden_ratio: Ratio to scale hidden dimension for intermediate size calculation.
        intermediate_dim: Optional explicit intermediate size.
    """

    hidden_ratio: int | float | None = None
    intermediate_dim: int | None = None
    use_bias: bool = False

    def build(self, in_features: int, out_features: int | None, key: PRNGKeyArray) -> SwiGLU:
        """Build SwiGLU from config.

        Args:
            in_features: Input dimensionality.
            out_features: Optional output dimensionality. If None, defaults to in_features.
            key: JAX random key for initialization.

        Returns:
            The SwiGLU instance.
        """
        # out_features is unused for SwiGLU since it keeps input dimension the same
        return SwiGLU(in_features=in_features, cfg=self, key=key)


class SwiGLU[ConfigType: SwiGLUConfig](ChannelMixer):
    """Swish Gated Linear Unit (SwiGLU) layer.

    Adapted from https://huggingface.co/blog/sachithgunasekara/nanojaxgpt .

    The architecture consists of three linear projections:
    - gate_proj: Projects input to intermediate dimension
    - up_proj: Projects input to intermediate dimension
    - down_proj: Projects intermediate dimension back to hidden dimension
    The computation is: down_proj(swish(gate_proj(x)) * up_proj(x))

    Args:
        in_features: Dimensionality of the input and output features.
        hidden_ratio: Ratio to scale hidden dimension for intermediate size calculation.
            If None, defaults to 4.
        intermediate_dim: Dimensionality of the intermediate projection.
            If None, calculated as `int(hidden_dim * hidden_ratio * 2/3)`
            rounded to nearest multiple of 256.
        key: JAX random key for weight initialization.
    """

    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        *,
        out_features: int | None = None,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)

        hidden_ratio = 4 if cfg.hidden_ratio is None else cfg.hidden_ratio
        intermediate_dim = cfg.intermediate_dim
        if intermediate_dim is None:
            intermediate_dim = int(in_features * hidden_ratio * 2 / 3)
            intermediate_dim = 256 * ((intermediate_dim + 256 - 1) // 256)

        self.gate_proj = eqx.nn.Linear(
            in_features, intermediate_dim, use_bias=cfg.use_bias, key=k1
        )
        self.up_proj = eqx.nn.Linear(in_features, intermediate_dim, use_bias=cfg.use_bias, key=k2)
        self.down_proj = eqx.nn.Linear(
            intermediate_dim, in_features, use_bias=cfg.use_bias, key=k3
        )

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
