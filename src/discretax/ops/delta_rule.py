"""Chunked delta rule operator.

Implements the hardware-efficient chunked form of the delta rule recurrence used by DeltaNet.

References:
    Yang et al. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length.
    https://arxiv.org/abs/2406.06484
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray  # noqa: F401


def chunk_delta_rule_head(
    Q: Array,
    K: Array,
    V: Array,
    beta: Array,
    chunk_size: int,
) -> Array:
    """Apply the chunked delta rule to a single attention head.

    Computes the delta rule recurrence efficiently by dividing the sequence into
    fixed-size chunks, using forward substitution within each chunk and
    `jax.lax.scan` across chunks.

    Args:
        Q: Query matrix for one head, shape (timesteps, head_dim).
        K: Key matrix for one head, shape (timesteps, head_dim).
        V: Value matrix for one head, shape (timesteps, head_dim).
        beta: Per-timestep scalar gate, shape (timesteps,). Should be in (0, 1).
        chunk_size: Number of timesteps per chunk. Must evenly divide the sequence length.

    Returns:
        Output sequence for this head, shape (timesteps, head_dim).
    """
    _, hidden_dim = Q.shape

    Q = einops.rearrange(Q, "(n c) d -> n c d", c=chunk_size)
    K = einops.rearrange(K, "(n c) d -> n c d", c=chunk_size)
    V = einops.rearrange(V, "(n c) d -> n c d", c=chunk_size)
    beta = einops.rearrange(beta, "(n c) -> n c", c=chunk_size)

    K_beta = K * beta[..., None]
    V_beta = V * beta[..., None]

    # Build lower-triangular S and invert (I + S) via forward substitution.
    S = -jnp.tril(K_beta @ K.mT, -1)
    for i in range(1, chunk_size):
        S = S.at[..., i, :i].set(S[..., i, :i] + (S[..., i, :, None] * S[..., :, :i]).sum(axis=-2))
    T = S + jnp.eye(chunk_size)

    U = T @ V_beta
    W = T @ K_beta

    chunk_H = jnp.zeros((hidden_dim, hidden_dim))
    _, O = jax.lax.scan(_chunk_scan, chunk_H, (Q, K, U, W))

    return einops.rearrange(O, "n c d -> (n c) d")


def _chunk_scan(carry: Array, x: tuple) -> tuple:
    """Single-chunk scan step for the delta rule.

    Args:
        carry: Hidden state matrix H of shape (head_dim, head_dim).
        x: Tuple of (Q, K, U, W) chunk slices, each (chunk_size, head_dim).

    Returns:
        Tuple of (updated H, chunk output O).
    """
    q, k, u, w = x
    h = carry
    pseudo_v = u - w @ h.T
    o = q @ h.T + jnp.tril(q @ k.T, 0) @ pseudo_v
    h = h + pseudo_v.T @ k
    return h, o


def chunk_delta_rule(
    Q: Array,
    K: Array,
    V: Array,
    beta: Array,
    chunk_size: int,
) -> Array:
    """Apply the chunked delta rule across all attention heads.

    Vectorises `chunk_delta_rule_head` over the heads dimension.

    Args:
        Q: Query tensor, shape (timesteps, heads, head_dim).
        K: Key tensor, shape (timesteps, heads, head_dim).
        V: Value tensor, shape (timesteps, heads, head_dim).
        beta: Per-timestep scalar gate, shape (timesteps,).
        chunk_size: Number of timesteps per chunk. Must evenly divide timesteps.

    Returns:
        Output tensor, shape (timesteps, heads, head_dim).
    """
    return jax.vmap(chunk_delta_rule_head, in_axes=(1, 1, 1, None, None), out_axes=1)(
        Q, K, V, beta, chunk_size
    )
