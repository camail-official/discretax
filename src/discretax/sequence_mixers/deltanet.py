"""Sequence mixer for DeltaNet.

DeltaNet is a linear attention model that uses the delta rule for hidden state updates,
offering an efficient alternative to softmax attention with linear complexity in sequence length.

References:
    Yang et al. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length.
    https://arxiv.org/abs/2406.06484
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.ops.delta_rule import chunk_delta_rule
from discretax.sequence_mixers.base import AbstractSequenceMixer


class DeltaNetSequenceMixer(AbstractSequenceMixer):
    """DeltaNet sequence mixer layer.

    Implements multi-head linear attention with delta rule updates.
    Input is projected to queries, keys, values and a scalar gate (beta),
    the chunked delta rule recurrence is applied per head, and the result
    is projected back to the input dimension.

    Attributes:
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        beta_proj: Scalar gate projection (sigmoid-activated at call time).
        out_proj: Output projection back to in_features.
        n_heads: Number of attention heads.
        head_dim: Dimensionality of each head.
        chunk_size: Number of timesteps per chunk for the chunked delta rule.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    beta_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        n_heads: int = 4,
        head_dim: int | None = None,
        chunk_size: int = 64,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the DeltaNet sequence mixer.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.
            n_heads: Number of attention heads.
            head_dim: Dimensionality per head. Defaults to `in_features // n_heads`.
            chunk_size: Timesteps per chunk for the chunked delta rule. Must evenly
                divide the sequence length at call time.
            use_bias: Whether to include bias terms in the projections.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        if head_dim is None:
            head_dim = in_features // n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        inner_dim = n_heads * head_dim
        k_q, k_k, k_v, k_beta, k_out = jr.split(key, 5)

        self.q_proj = eqx.nn.Linear(in_features, inner_dim, use_bias=False, key=k_q)
        self.k_proj = eqx.nn.Linear(in_features, inner_dim, use_bias=False, key=k_k)
        self.v_proj = eqx.nn.Linear(in_features, inner_dim, use_bias=False, key=k_v)
        self.beta_proj = eqx.nn.Linear(in_features, 1, use_bias=use_bias, key=k_beta)
        self.out_proj = eqx.nn.Linear(inner_dim, in_features, use_bias=use_bias, key=k_out)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the DeltaNet sequence mixer.

        Args:
            x: Input sequence of shape (timesteps, in_features). The sequence length
                must be divisible by `chunk_size`.
            key: JAX random key (unused, kept for interface compatibility).

        Returns:
            Output sequence of shape (timesteps, in_features).
        """
        L, _ = x.shape
        assert L % self.chunk_size == 0, (
            f"Sequence length {L} must be divisible by chunk_size {self.chunk_size}."
        )

        Q = jax.vmap(self.q_proj)(x)  # (L, n_heads * head_dim)
        K = jax.vmap(self.k_proj)(x)
        V = jax.vmap(self.v_proj)(x)
        beta = jax.nn.sigmoid(jax.vmap(self.beta_proj)(x)[:, 0])  # (L,)

        Q = Q.reshape(L, self.n_heads, self.head_dim)
        K = K.reshape(L, self.n_heads, self.head_dim)
        V = V.reshape(L, self.n_heads, self.head_dim)

        # L2-normalise Q and K for linear attention stability.
        Q = Q / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + 1e-6)
        K = K / (jnp.linalg.norm(K, axis=-1, keepdims=True) + 1e-6)

        O = chunk_delta_rule(Q, K, V, beta, self.chunk_size)  # (L, n_heads, head_dim)

        O = O.reshape(L, self.n_heads * self.head_dim)
        return jax.vmap(self.out_proj)(O)

    def __repr__(self) -> str:
        """Return a string representation of the DeltaNet sequence mixer.

        Returns:
            Compact summary showing key dimensions.
        """
        in_features = self.q_proj.in_features
        return (
            f"DeltaNetSequenceMixer({in_features}, "
            f"n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, "
            f"chunk_size={self.chunk_size})"
        )
