"""Sequence Mixer for S5 (Simplified State Space Layers).

S5 implementation modified from: https://github.com/lindermanlab/S5/blob/main/s5/ssm_init.py

See: https://arxiv.org/abs/2208.04933
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag
from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixer, SequenceMixerConfig


@dataclass(frozen=True)
class S5SequenceMixerConfig(SequenceMixerConfig):
    """Configuration for the S5 sequence mixer.

    This configuration class defines the hyperparameters for the S5 sequence mixer.
    S5 uses structured state space models with HiPPO initialization for efficient
    sequence modeling.

    Attributes:
        state_dim: Dimensionality of the state space (total SSM size).
        ssm_blocks: Number of SSM blocks (for block-diagonal structure).
        C_init: Initialization method for output matrix C.
        conj_sym: Whether to enforce conjugate symmetry (reduces parameters by half).
        clip_eigs: Whether to clip eigenvalues to ensure stability.
        discretization: Discretization method to use.
        dt_min: Minimum discretization step size.
        dt_max: Maximum discretization step size.
        step_rescale: Rescaling factor for the discretization step.
    """

    state_dim: int = 64
    ssm_blocks: int = 1
    C_init: Literal["trunc_standard_normal", "lecun_normal", "complex_normal"] = "lecun_normal"
    conj_sym: bool = True
    clip_eigs: bool = True
    discretization: Literal["zoh", "bilinear"] = "zoh"
    dt_min: float = 0.001
    dt_max: float = 1.0
    step_rescale: float = 1.0

    def build(self, in_features: int, key: PRNGKeyArray) -> S5SequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """
        return S5SequenceMixer(in_features=in_features, cfg=self, key=key)


class S5SequenceMixer[ConfigType: S5SequenceMixerConfig](SequenceMixer):
    """S5 sequence mixer layer.

    This layer implements the Simplified State Space Layers (S5) sequence mixer,
    which uses structured state space models with HiPPO initialization and efficient
    parallel scan operations.

    Attributes:
        Lambda_re: Real part of diagonal state matrix eigenvalues.
        Lambda_im: Imaginary part of diagonal state matrix eigenvalues.
        B: Input projection matrix (parameterized as V^{-1}B).
        C: Output projection matrix (parameterized as CV).
        D: Skip connection weights.
        log_step: Log of discretization step sizes.
        H: Number of hidden channels (input features).
        P: Effective state dimensionality.
        conj_sym: Whether conjugate symmetry is enforced.
        clip_eigs: Whether to clip eigenvalues for stability.
        discretization: Discretization method being used.
        step_rescale: Rescaling factor for step sizes.
    """

    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    log_step: jax.Array

    H: int
    P: int
    conj_sym: bool
    clip_eigs: bool
    discretization: str
    step_rescale: float

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """Initialize the S5 sequence mixer layer.

        Args:
            in_features: Input dimensionality.
            cfg: Configuration for the S5 sequence mixer.
            key: JAX random key for initialization.
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        B_key, C_key, D_key, step_key, key = jr.split(key, 5)

        ssm_size = cfg.state_dim
        blocks = cfg.ssm_blocks
        C_init = cfg.C_init
        conj_sym = cfg.conj_sym
        clip_eigs = cfg.clip_eigs
        discretization = cfg.discretization
        dt_min = cfg.dt_min
        dt_max = cfg.dt_max
        step_rescale = cfg.step_rescale

        block_size = int(ssm_size / blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = _make_dplr_hippo(block_size)

        if conj_sym:
            block_size = block_size // 2
            P = ssm_size // 2
        else:
            P = ssm_size

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        self.in_features = in_features
        self.P = P
        if conj_sym:
            local_P = 2 * P
        else:
            local_P = P

        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag

        self.conj_sym = conj_sym

        self.clip_eigs = clip_eigs

        self.B = _init_vinb(lecun_normal(), B_key, (local_P, self.in_features), Vinv)

        # Initialize state to output (C) matrix
        if C_init in ["trunc_standard_normal"]:
            C_init_fn = _trunc_standard_normal
        elif C_init in ["lecun_normal"]:
            C_init_fn = lecun_normal()
        elif C_init in ["complex_normal"]:
            C_init_fn = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError(f"C_init method {C_init} not implemented")

        if C_init in ["complex_normal"]:
            self.C = C_init_fn(C_key, (self.in_features, 2 * self.P, 2))
        else:
            self.C = _init_cv(C_init_fn, C_key, (self.in_features, local_P, 2), V)

        self.D = normal(stddev=1.0)(D_key, (self.in_features,))

        # Initialize learnable discretization timescale value
        self.log_step = _init_log_steps(step_key, (self.P, dt_min, dt_max))

        self.step_rescale = step_rescale
        self.discretization = discretization

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the S5 sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key (unused, for compatibility).

        Returns:
            The output of the S5 sequence mixer.
        """
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            Lambda_bar, B_bar = _discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            Lambda_bar, B_bar = _discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                f"Discretization method {self.discretization} not implemented"
            )

        ys = _apply_ssm(Lambda_bar, B_bar, C_tilde, x, self.conj_sym)

        # Add feedthrough matrix output Du
        Du = jax.vmap(lambda u: self.D * u)(x)
        return ys + Du


def _make_hippo(N: int) -> Array:
    """Create a HiPPO-LegS matrix.

    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py

    Args:
        N: State size.

    Returns:
        N x N HiPPO LegS matrix.
    """
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def _make_nplr_hippo(N: int) -> tuple[Array, Array, Array]:
    """Make components needed for NPLR representation of HiPPO-LegS.

    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py

    Args:
        N: State size.

    Returns:
        Tuple of (HiPPO matrix, low-rank factor P, input matrix B).
    """
    # Make -HiPPO
    hippo = _make_hippo(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def _make_dplr_hippo(N: int) -> tuple[Array, Array, Array, Array, Array]:
    """Make components needed for DPLR representation of HiPPO-LegS.

    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note: we will only use the diagonal part.

    Args:
        N: State size.

    Returns:
        Tuple of (eigenvalues Lambda, low-rank term P, conjugated input matrix B,
        eigenvectors V, original B matrix).
    """
    A, P, B = _make_nplr_hippo(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def _log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1):
    """Initialize the learnable timescale Delta by sampling uniformly.

    Args:
        dt_min: Minimum discretization step value.
        dt_max: Maximum discretization step value.

    Returns:
        Initialization function that samples log_step uniformly.
    """

    def init(key: PRNGKeyArray, shape: tuple) -> Array:
        """Initialize log_step values.

        Args:
            key: JAX random key.
            shape: Desired shape.

        Returns:
            Sampled log_step values.
        """
        return jr.uniform(key, shape) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)

    return init


def _init_log_steps(key: PRNGKeyArray, input: tuple) -> Array:
    """Initialize an array of learnable timescale parameters.

    Args:
        key: JAX random key.
        input: Tuple containing (dim, dt_min, dt_max) where dim is the shape.

    Returns:
        Initialized array of timescales of shape (dim,).
    """
    dim, dt_min, dt_max = input
    log_steps = []
    for i in range(dim):
        key, skey = jr.split(key)
        log_step = _log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return jnp.array(log_steps)


def _init_vinb(init_fun, rng: PRNGKeyArray, shape: tuple, Vinv: Array) -> Array:
    """Initialize B_tilde = V^{-1}B.

    First samples B, then computes V^{-1}B. Parameterized with two different
    matrices for complex numbers.

    Args:
        init_fun: Initialization function to use (e.g. lecun_normal()).
        rng: JAX random key.
        shape: Desired shape (P, H).
        Vinv: Inverse eigenvectors for initialization.

    Returns:
        B_tilde of shape (P, H, 2) for complex parameterization.
    """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def _trunc_standard_normal(key: PRNGKeyArray, shape: tuple) -> Array:
    """Sample C with a truncated normal distribution.

    Args:
        key: JAX random key.
        shape: Desired shape of length 3, (H, P, _).

    Returns:
        Sampled C matrix of shape (H, P, 2) for complex parameterization.
    """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = jr.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


def _init_cv(init_fun, rng: PRNGKeyArray, shape: tuple, V: Array) -> Array:
    """Initialize C_tilde = CV.

    First samples C, then computes CV. Parameterized with two different
    matrices for complex numbers.

    Args:
        init_fun: Initialization function to use (e.g. lecun_normal()).
        rng: JAX random key.
        shape: Desired shape (H, P).
        V: Eigenvectors for initialization.

    Returns:
        C_tilde of shape (H, P, 2) for complex parameterization.
    """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


def _discretize_bilinear(Lambda: Array, B_tilde: Array, Delta: Array) -> tuple[Array, Array]:
    """Discretize a diagonalized continuous-time linear SSM using bilinear transform.

    Args:
        Lambda: Diagonal state matrix (P,).
        B_tilde: Input matrix (P, H).
        Delta: Discretization step sizes (P,).

    Returns:
        Tuple of (discretized Lambda_bar, discretized B_bar).
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def _discretize_zoh(Lambda: Array, B_tilde: Array, Delta: Array) -> tuple[Array, Array]:
    """Discretize a diagonalized continuous-time linear SSM using zero-order hold.

    Args:
        Lambda: Diagonal state matrix (P,).
        B_tilde: Input matrix (P, H).
        Delta: Discretization step sizes (P,).

    Returns:
        Tuple of (discretized Lambda_bar, discretized B_bar).
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


@jax.vmap
def _binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.

    Assumes a diagonal matrix A.

    Args:
        q_i: Tuple containing (A_i, Bu_i) at position i.
        q_j: Tuple containing (A_j, Bu_j) at position j.

    Returns:
        New element (A_out, Bu_out).
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def _apply_ssm(
    Lambda_bar: Array,
    B_bar: Array,
    C_tilde: Array,
    input_sequence: Array,
    conj_sym: bool,
) -> Array:
    """Compute the L x H output of discretized SSM given an L x H input.

    Args:
        Lambda_bar: Discretized diagonal state matrix (P,).
        B_bar: Discretized input matrix (P, H).
        C_tilde: Output matrix (H, P).
        input_sequence: Input sequence of features (L, H).
        conj_sym: Whether conjugate symmetry is enforced.

    Returns:
        The SSM outputs (S5 layer preactivations) of shape (L, H).
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(_binary_operator, (Lambda_elements, Bu_elements))

    if conj_sym:
        return jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)
