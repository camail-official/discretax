"""Sequence Mixer for LRU (Linear Recurrent Unit).

Adapted from https://github.com/tk-rusch/linoss/blob/main/models/LRU.py

See: https://arxiv.org/abs/2303.06349
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixer, SequenceMixerConfig


def _binary_operator_diag(element_i, element_j):
    """Binary operator for parallel scan of LRU recurrence.

    This function implements the binary operator for the parallel scan of the LRU recurrence.

    Args:
        element_i: Tuple containing (a_i, bu_i) at position i.
        element_j: Tuple containing (a_j, bu_j) at position j.

    Returns:
        The combined elements for the associative scan.
    """
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


@dataclass(frozen=True)
class LRUSequenceMixerConfig(SequenceMixerConfig):
    """Configuration for the LRU sequence mixer.

    This configuration class defines the hyperparameters for the LRU sequence mixer.
    LRU uses complex-valued diagonal state matrices for efficient sequence modeling.

    Attributes:
        state_dim: Dimensionality of the state space.
        r_min: Minimum radius for the complex-valued eigenvalues.
        r_max: Maximum radius for the complex-valued eigenvalues.
        max_phase: Maximum phase angle for the complex-valued eigenvalues.
    """

    state_dim: int = 64
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28

    def build(self, in_features: int, key: PRNGKeyArray) -> LRUSequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """
        return LRUSequenceMixer(in_features=in_features, cfg=self, key=key)


class LRUSequenceMixer[ConfigType: LRUSequenceMixerConfig](SequenceMixer):
    """LRU sequence mixer layer.

    This layer implements the Linear Recurrent Unit (LRU) sequence mixer using
    complex-valued diagonal state matrices for efficient and expressive sequence modeling.

    Attributes:
        nu_log: Log of nu parameter (controls eigenvalue magnitudes).
        theta_log: Log of theta parameter (controls eigenvalue phases).
        B_re: Real part of input projection matrix.
        B_im: Imaginary part of input projection matrix.
        C_re: Real part of output projection matrix.
        C_im: Imaginary part of output projection matrix.
        D: Skip connection weights.
        gamma_log: Log of normalization factor.

    Args:
        in_features: Input dimensionality.
        cfg: Configuration for the LRU sequence mixer.
        key: JAX random key for initialization.
        **kwargs: Additional keyword arguments (unused, for compatibility).
    """

    nu_log: jax.Array
    theta_log: jax.Array
    B_re: jax.Array
    B_im: jax.Array
    C_re: jax.Array
    C_im: jax.Array
    D: jax.Array
    gamma_log: jax.Array

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """Initialize the LRU sequence mixer layer."""
        u1_key, u2_key, B_re_key, B_im_key, C_re_key, C_im_key, D_key = jr.split(key, 7)

        N = cfg.state_dim
        H = in_features
        r_max = cfg.r_max
        r_min = cfg.r_min
        max_phase = cfg.max_phase

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = jr.uniform(u1_key, shape=(N,))
        u2 = jr.uniform(u2_key, shape=(N,))
        self.nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        self.theta_log = jnp.log(max_phase * u2)

        # Glorot initialized Input/Output projection matrices
        self.B_re = jr.normal(B_re_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.B_im = jr.normal(B_im_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.C_re = jr.normal(C_re_key, shape=(H, N)) / jnp.sqrt(N)
        self.C_im = jr.normal(C_im_key, shape=(H, N)) / jnp.sqrt(N)
        self.D = jr.normal(D_key, shape=(H,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LRU sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key (unused, for compatibility).

        Returns:
            The output of the LRU sequence mixer.
        """
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
        C = self.C_re + 1j * self.C_im
        # Running the LRU + output projection
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.associative_scan(_binary_operator_diag, elements)  # all x_k
        y = jax.vmap(lambda z, u: (C @ z).real + (self.D * u))(inner_states, x)

        return y
