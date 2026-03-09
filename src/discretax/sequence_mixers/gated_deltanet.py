"""Sequence mixer for GatedDeltaNet.

This module implements a gated-delta recurrence in JAX/Equinox. It keeps the
key ingredients of the original design:

- projected multi-head Q/K/V features,
- learnable per-head decay parameters (`A_log`, `dt_bias`),
- per-token update gate (`beta`),
- recurrent state update in key-value state space,
- optional output gating and RMS-style normalization before output projection.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.sequence_mixers.base import AbstractSequenceMixer


def _inverse_softplus(x: Array) -> Array:
    """Compute the inverse softplus transform.

    Args:
        x: Positive input values.

    Returns:
        Values `y` such that `softplus(y) ~= x`.
    """
    return x + jnp.log(-jnp.expm1(-x))


def _l2norm(x: Array, eps: float = 1e-6) -> Array:
    """Apply L2 normalization along the last axis.

    Args:
        x: Input tensor.
        eps: Numerical stabilizer.

    Returns:
        L2-normalized tensor with same shape as input.
    """
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _gated_delta_recurrent_step(
    h: Array, x: tuple[Array, Array, Array, Array, Array]
) -> tuple[Array, Array]:
    """Single recurrent update for gated delta rule.

    Args:
        h: Current state matrix, shape `(heads, k_dim, v_dim)`.
        x: Tuple `(q_t, k_t, v_t, g_t, beta_t)` for one timestep.

    Returns:
        Tuple of updated state and timestep output.
    """
    q_t, k_t, v_t, g_t, beta_t = x
    h = h * jnp.exp(g_t)[:, None, None]
    v_res = v_t - jnp.einsum("hkv,hk->hv", h, k_t)
    v_res = v_res * beta_t[:, None]
    h = h + jnp.einsum("hk,hv->hkv", k_t, v_res)
    o_t = jnp.einsum("hk,hkv->hv", q_t, h)
    return h, o_t


def gated_delta_recurrent_rule(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
) -> Array:
    """Apply gated-delta recurrence across a sequence.

    Args:
        q: Queries of shape `(timesteps, heads, k_dim)`.
        k: Keys of shape `(timesteps, heads, k_dim)`.
        v: Values of shape `(timesteps, heads, v_dim)`.
        g: Log-space decay gates of shape `(timesteps, heads)`.
        beta: Value update gates of shape `(timesteps, heads)`.

    Returns:
        Output tensor of shape `(timesteps, heads, v_dim)`.
    """
    _, n_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    h0 = jnp.zeros((n_heads, k_dim, v_dim), dtype=q.dtype)
    _, out = jax.lax.scan(_gated_delta_recurrent_step, h0, (q, k, v, g, beta))
    return out


class GatedDeltaNetSequenceMixer(AbstractSequenceMixer):
    """Gated DeltaNet sequence mixer.

    This follows the GatedDeltaNet core equations with JAX-native recurrent updates.

    Attributes:
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        a_proj: Projection used to parameterize decay gate preactivations.
        b_proj: Projection used to parameterize update gate `beta`.
        g_proj: Optional output gate projection.
        o_proj: Output projection back to `in_features`.
        A_log: Learnable per-head decay-rate parameter (log-space).
        dt_bias: Learnable per-head decay bias in softplus parameterization.
        n_heads: Number of key/query heads.
        num_v_heads: Number of value heads.
        head_k_dim: Per-head key/query dimension.
        head_v_dim: Per-head value dimension.
        chunk_size: Stored chunk size configuration.
        mode: Recurrence mode (`"chunk"` or `"fused_recurrent"`).
        use_gate: Whether to use output gating.
        allow_neg_eigval: Whether to scale `beta` by 2.
        norm_eps: Epsilon for output RMS normalization.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    a_proj: eqx.nn.Linear
    b_proj: eqx.nn.Linear
    g_proj: eqx.nn.Linear | None
    o_proj: eqx.nn.Linear

    A_log: Array
    dt_bias: Array

    n_heads: int = eqx.field(static=True)
    num_v_heads: int = eqx.field(static=True)
    head_k_dim: int = eqx.field(static=True)
    head_v_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    use_gate: bool = eqx.field(static=True)
    allow_neg_eigval: bool = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        n_heads: int = 4,
        num_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        chunk_size: int = 64,
        mode: str = "chunk",
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        """Initialize the GatedDeltaNet sequence mixer.

        Args:
            in_features: Input feature dimension.
            key: JAX PRNG key.
            n_heads: Number of key/query heads.
            num_v_heads: Number of value heads. Defaults to `n_heads`.
            head_dim: Key/query head dimension. Defaults to `in_features // n_heads`.
            expand_v: Value expansion factor (`head_v_dim = int(head_dim * expand_v)`).
            chunk_size: Configured chunk size for chunked mode.
            mode: Execution mode (`"chunk"` or `"fused_recurrent"`).
            use_gate: Whether to apply output gating.
            allow_neg_eigval: Whether to scale beta by 2.
            norm_eps: Epsilon used in RMS normalization.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        del args, kwargs
        if head_dim is None:
            head_dim = in_features // n_heads
        if num_v_heads is None:
            num_v_heads = n_heads
        if num_v_heads % n_heads != 0:
            raise ValueError(f"num_v_heads={num_v_heads} must be divisible by n_heads={n_heads}.")
        if mode not in ("chunk", "fused_recurrent"):
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'chunk' or 'fused_recurrent'.")

        head_v_dim = int(head_dim * expand_v)
        if float(head_v_dim) != head_dim * expand_v:
            raise ValueError(f"head_dim * expand_v must be integer, got {head_dim * expand_v}.")

        self.n_heads = n_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_v_dim
        self.chunk_size = chunk_size
        self.mode = mode
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps

        key_dim = n_heads * head_dim
        value_dim = num_v_heads * head_v_dim
        keys = jr.split(key, 9)

        self.q_proj = eqx.nn.Linear(in_features, key_dim, use_bias=False, key=keys[0])
        self.k_proj = eqx.nn.Linear(in_features, key_dim, use_bias=False, key=keys[1])
        self.v_proj = eqx.nn.Linear(in_features, value_dim, use_bias=False, key=keys[2])
        self.a_proj = eqx.nn.Linear(in_features, num_v_heads, use_bias=False, key=keys[3])
        self.b_proj = eqx.nn.Linear(in_features, num_v_heads, use_bias=False, key=keys[4])
        self.g_proj = (
            eqx.nn.Linear(in_features, value_dim, use_bias=False, key=keys[5])
            if use_gate
            else None
        )
        self.o_proj = eqx.nn.Linear(value_dim, in_features, use_bias=False, key=keys[6])

        A = jr.uniform(keys[7], (num_v_heads,), minval=0.0, maxval=16.0)
        self.A_log = jnp.log(A)
        dt = jr.uniform(keys[8], (num_v_heads,), minval=0.001, maxval=0.1)
        self.dt_bias = _inverse_softplus(dt)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the GatedDeltaNet sequence mixer.

        Args:
            x: Input sequence of shape `(timesteps, in_features)`.
            key: JAX random key (unused, kept for interface compatibility).

        Returns:
            Output sequence of shape `(timesteps, in_features)`.
        """
        del key
        l, _ = x.shape

        q = jax.vmap(self.q_proj)(x).reshape(l, self.n_heads, self.head_k_dim)
        k = jax.vmap(self.k_proj)(x).reshape(l, self.n_heads, self.head_k_dim)
        v = jax.vmap(self.v_proj)(x).reshape(l, self.num_v_heads, self.head_v_dim)

        q = _l2norm(q)
        k = _l2norm(k)

        if self.num_v_heads > self.n_heads:
            groups = self.num_v_heads // self.n_heads
            q = jnp.repeat(q, repeats=groups, axis=1)
            k = jnp.repeat(k, repeats=groups, axis=1)

        beta = jax.nn.sigmoid(jax.vmap(self.b_proj)(x))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        a = jax.vmap(self.a_proj)(x)
        g = -jnp.exp(self.A_log)[None, :] * jax.nn.softplus(a + self.dt_bias[None, :])

        scale = jnp.sqrt(float(self.head_k_dim))
        q = q / scale

        if self.mode == "chunk":
            o = gated_delta_recurrent_rule(q, k, v, g, beta)
        else:
            o = gated_delta_recurrent_rule(q, k, v, g, beta)

        if self.use_gate:
            if self.g_proj is None:
                raise ValueError("g_proj is None while use_gate=True")
            gate = jax.vmap(self.g_proj)(x).reshape(l, self.num_v_heads, self.head_v_dim)
            rms = jnp.sqrt(jnp.mean(jnp.square(o), axis=-1, keepdims=True) + self.norm_eps)
            o = (o / rms) * jax.nn.silu(gate)
        else:
            rms = jnp.sqrt(jnp.mean(jnp.square(o), axis=-1, keepdims=True) + self.norm_eps)
            o = o / rms

        o = o.reshape(l, self.num_v_heads * self.head_v_dim)
        return jax.vmap(self.o_proj)(o)

    def __repr__(self) -> str:
        """Return a compact representation of the mixer configuration."""
        in_features = self.q_proj.in_features
        return (
            f"GatedDeltaNetSequenceMixer({in_features}, n_heads={self.n_heads}, "
            f"num_v_heads={self.num_v_heads}, head_dim={self.head_k_dim}, "
            f"expand_v={self.head_v_dim / self.head_k_dim:.2f}, mode='{self.mode}')"
        )
