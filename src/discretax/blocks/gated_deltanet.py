"""GatedDeltaNet block with pre-norm residual wiring.

This block implements:

1. attention/sequence branch with pre-norm and residual add,
2. MLP/channel branch with pre-norm and residual add.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.base import AbstractBlock
from discretax.channel_mixers.base import AbstractChannelMixer
from discretax.sequence_mixers.base import AbstractSequenceMixer
from discretax.utils.config_mixin import Resolvable


class _RMSNorm(eqx.Module):
    """Minimal RMSNorm module.

    This lightweight implementation is local to the GatedDeltaNet block to
    reproduce RMS pre-normalization.
    """

    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Numerical stabilizer.
        """
        self.weight = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape `(..., dim)`.

        Returns:
            Normalized tensor with the same shape.
        """
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class GatedDeltaNetBlock(AbstractBlock):
    """A block matching GatedDeltaNet pre-norm residual structure.

    Attributes:
        attn_norm: RMS normalization before sequence mixer.
        mlp_norm: RMS normalization before channel mixer.
        sequence_mixer: Sequence mixer module (typically `GatedDeltaNetSequenceMixer`).
        channel_mixer: Channel mixer module (typically `SwiGLU`).
        drop: Dropout applied to both residual branches.
    """

    attn_norm: _RMSNorm
    mlp_norm: _RMSNorm
    sequence_mixer: AbstractSequenceMixer
    channel_mixer: AbstractChannelMixer
    drop: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        sequence_mixer: Resolvable[AbstractSequenceMixer],
        channel_mixer: Resolvable[AbstractChannelMixer],
        drop_rate: float = 0.0,
        norm_eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize a GatedDeltaNet block.

        Args:
            in_features: Input feature dimension.
            key: JAX PRNG key.
            sequence_mixer: Sequence mixer or partial resolver.
            channel_mixer: Channel mixer or partial resolver.
            drop_rate: Dropout probability for each branch.
            norm_eps: Epsilon for RMS normalization.
            *args: Additional positional arguments (ignored).
            **kwargs: Extra kwargs forwarded to mixer resolvers.
        """
        del args
        k_seq, k_chan, k_drop = jr.split(key, 3)
        self.attn_norm = _RMSNorm(in_features, eps=norm_eps)
        self.mlp_norm = _RMSNorm(in_features, eps=norm_eps)
        self.sequence_mixer = sequence_mixer.resolve(in_features=in_features, key=k_seq, **kwargs)
        self.channel_mixer = channel_mixer.resolve(in_features=in_features, key=k_chan, **kwargs)
        self.drop = eqx.nn.Dropout(drop_rate, inference=False)

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Apply one GatedDeltaNet block.

        Args:
            x: Input sequence of shape `(timesteps, in_features)`.
            state: Equinox state container, passed through unchanged.
            key: JAX PRNG key for dropout.

        Returns:
            Tuple of `(output, state)`.
        """
        k1, k2 = jr.split(key, 2)

        residual = x
        y = self.attn_norm(x)
        y = self.sequence_mixer(y, k1)
        y = self.drop(y, key=k1)
        x = residual + y

        residual = x
        y = self.mlp_norm(x)
        y = jax.vmap(self.channel_mixer)(y)
        y = self.drop(y, key=k2)
        x = residual + y

        return x, state
