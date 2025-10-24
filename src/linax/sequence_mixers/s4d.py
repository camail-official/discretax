"""Sequence Mixer for S4D (Structured State Space - Diagonal).

See: https://arxiv.org/abs/2206.11893
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixer, SequenceMixerConfig


@dataclass(frozen=True)
class S4DSequenceMixerConfig(SequenceMixerConfig):
    """Configuration for the S4D sequence mixer.

    This configuration class defines the hyperparameters for the S4D sequence mixer.
    S4D uses diagonal structured state space models with efficient FFT-based convolutions.

    Attributes:
        state_dim: Dimensionality of the state space.
        transposed: Whether input is in transposed format (H, L) vs (L, H).
        dt_min: Minimum discretization step size.
        dt_max: Maximum discretization step size.
    """

    state_dim: int = 64
    transposed: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1

    def build(self, in_features: int, key: PRNGKeyArray) -> S4DSequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """
        return S4DSequenceMixer(in_features=in_features, cfg=self, key=key)


class S4DSequenceMixer[ConfigType: S4DSequenceMixerConfig](SequenceMixer):
    """S4D sequence mixer layer.

    This layer implements the Structured State Space - Diagonal (S4D) sequence mixer,
    which uses diagonal parameterization of state space models for efficient sequence modeling
    via FFT-based convolutions.

    Attributes:
        in_features: Input dimensionality.
        state_dim: State space dimensionality.
        transposed: Whether input is in transposed format.
        kernel: The S4D kernel for generating convolution kernels.

    Args:
        in_features: Input dimensionality.
        cfg: Configuration for the S4D sequence mixer.
        key: JAX random key for initialization.
        **kwargs: Additional keyword arguments (unused, for compatibility).
    """

    in_features: int
    state_dim: int
    transposed: bool
    kernel: _S4DKernel

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """Initialize the S4D sequence mixer layer."""
        self.in_features = in_features
        self.state_dim = cfg.state_dim
        self.transposed = cfg.transposed
        (k_kernel,) = jax.random.split(key, 1)
        self.kernel = _S4DKernel(
            self.in_features,
            N=self.state_dim,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            key=k_kernel,
        )

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the S4D sequence mixer layer.

        Args:
            x: Input sequence of features with shape (L, H) where L is sequence length
               and H is the number of hidden features.
            key: JAX random key (unused, for compatibility).

        Returns:
            The output of the S4D sequence mixer with shape (L, H).
        """
        x = x.T  # (time, hidden) -> (hidden, time)

        H, L = x.shape
        assert H == self.in_features, f"channel mismatch: got {H}, expected {self.in_features}"

        # Kernel: (hidden, time)
        k = self.kernel(L)

        # FFT-based linear convolution with zero-padding to 2L
        n_fft = 2 * L
        k_f = jnp.fft.rfft(k, n=n_fft, axis=-1)  # (hidden, n_fft//2+1)
        x_f = jnp.fft.rfft(x, n=n_fft, axis=-1)  # (hidden, n_fft//2+1)

        y = jnp.fft.irfft(x_f * k_f, n=n_fft, axis=-1)[..., :L]  # (hidden, time)

        return y.T  # return (time, hidden)


class _S4DKernel(eqx.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    This kernel implements the diagonal parameterization of structured state space models,
    which allows for efficient computation using FFT-based convolutions.

    Attributes:
        C: Complex-valued output projection matrix (H, N//2).
        log_dt: Log of discretization step sizes (H,).
        log_a_real: Log of real part of diagonal state matrix (H, N//2).
        A_imag: Imaginary part of diagonal state matrix (H, N//2).
    """

    C: jax.Array  # (H, N//2) complex
    log_dt: jax.Array  # (H,)
    log_a_real: jax.Array  # (H, N//2)
    A_imag: jax.Array  # (H, N//2)

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the S4D kernel.

        Args:
            d_model: Model dimensionality (number of channels).
            N: State space dimensionality.
            dt_min: Minimum discretization step size.
            dt_max: Maximum discretization step size.
            key: JAX random key for initialization.
        """
        H = d_model
        k1, k2, k3 = jax.random.split(key, 3)

        # Generate log_dt ~ Uniform(log(dt_min), log(dt_max))
        log_dt = jax.random.uniform(
            k1,
            shape=(H,),
            minval=math.log(dt_min),
            maxval=math.log(dt_max),
        )

        # C ~ Normal(0,1), complex
        real = jax.random.normal(k2, (H, N // 2))
        imag = jax.random.normal(k3, (H, N // 2))
        C = real + 1j * imag  # (H, N//2)

        # A parameters
        log_a_real = jnp.log(0.5 * jnp.ones((H, N // 2)))
        A_imag = math.pi * jnp.tile(jnp.arange(N // 2)[None, :], (H, 1))

        self.C = C
        self.log_dt = log_dt
        self.log_a_real = log_a_real
        self.A_imag = A_imag

    def __call__(self, L: int) -> Array:
        """Generate the convolution kernel for a given sequence length.

        Args:
            L: Sequence length.

        Returns:
            Convolution kernel of shape (H, L) where H is the number of channels.
        """
        # Materialize parameters
        dt = jnp.exp(self.log_dt)  # (H,)
        C = self.C  # (H, N//2)
        A = -jnp.exp(self.log_a_real) + 1j * self.A_imag  # (H, N//2)

        # Vandermonde multiplication
        dtA = A * dt[:, None]  # (H, N//2)
        t = jnp.arange(L)  # (L,)
        K = dtA[:, :, None] * t[None, None, :]  # (H, N//2, L)

        # Adjust C
        C = C * (jnp.exp(dtA) - 1.0) / A  # (H, N//2)

        # Perform contraction: 2 * sum_n C(h,n) * exp(K(h,n,l))
        K = 2 * jnp.einsum("hn,hnl->hl", C, jnp.exp(K)).real

        return K
