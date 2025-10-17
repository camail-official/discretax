"""Sequence Mixer for LinOSS-IM, LinOSS-IMEX, and Damped LinOSS-IMEX models.

See: https://arxiv.org/pdf/2410.03943
"""

import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray


def simple_uniform_init(rng, shape, std=1.0):
    """Simple uniform initialization."""
    weights = random.uniform(rng, shape) * 2.0 * std - std
    return weights


def map_theta_to_A(thetas, G_diag, steps):  # noqa: N802
    """Map theta to A."""
    A_plus = (
        4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))
    A_minus = (
        -4
        * jnp.sqrt(
            steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2)
        )
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))

    A_diag = jnp.where(thetas > jnp.pi / 2, A_plus, A_minus)

    return A_diag


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.

    Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def make_linoss_im_recurrence(A_diag, step):
    r"""Compute the PxP recurrent matrix M for LinOSS-IM.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        M    (float32): the recurrent matrix (P, P)
    """
    S = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = jnp.diag(1.0 - step**2.0 * A_diag * S)
    M_12 = jnp.diag(-1.0 * step * A_diag * S)
    M_21 = jnp.diag(step * S)
    M_22 = jnp.diag(S)

    M = jnp.block([[M_11, M_12], [M_21, M_22]])

    return M


def make_linoss_imex_recurrence(A_diag, step):
    r"""Compute the PxP recurrent matrix M for LinOSS-IMEX.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        M  (float32): the recurrent matrix (P, P)
    """
    M_11 = jnp.diag(jnp.ones_like(A_diag))
    M_12 = jnp.diag(-1.0 * step * A_diag)
    M_21 = jnp.diag(step)
    M_22 = jnp.diag(1.0 - (step**2.0) * A_diag)

    M = jnp.block([[M_11, M_12], [M_21, M_22]])

    return M


def make_damped_linoss_imex_recurrence(A_diag, G_diag, step):
    r"""Compute the PxP recurrent matrix M for Damped LinOSS-IMEX.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        G_diag  (float32):   diagonal damping matrix   (P,)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        M    (float32): the recurrent matrix (P, P)
    """
    I = jnp.ones_like(A_diag)
    S = I + step * G_diag
    M_11 = jnp.diag(1.0 / S)
    M_12 = jnp.diag(-step / S * A_diag)
    M_21 = jnp.diag(step / S)
    M_22 = jnp.diag(I - step**2 / S * A_diag)

    M = jnp.block([[M_11, M_12], [M_21, M_22]])

    return M


def apply_linoss_im(A_diag, B, input_sequence, step):
    r"""Compute the LxH output of LinOSS-IM given an LxH input.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        ys (float32): the SequenceMixer states (IM layer pre-output pre-activations) (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_12 = -1.0 * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = jnp.concatenate([M_11, M_12, M_21, M_22])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = M_11 * Bu_elements * step
    F2 = M_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def apply_linoss_imex(A_diag, B, input_sequence, step):
    r"""Compute the LxH output of of LinOSS-IMEX given an LxH input.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        ys (float32): the SequenceMixer states (IMEX layer pre-output pre-activations) (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    M = jnp.concatenate([A_, B_, C_, D_])

    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def apply_damped_linoss_imex(A_diag, G_diag, B, input_sequence, step):
    r"""Compute the LxH output of of Damped LinOSS-IMEX given an LxH input.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        G_diag  (float32):   diagonal damping matrix (P,)
        B       (complex64): input matrix            (P, H)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        ys (float32): the SequenceMixer states (Damped layer pre-output pre-activations) (L, P)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = Identity - step**2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


@dataclass
class LinOSSSequenceMixerConfig:
    """LinOSS configuration."""

    hidden_dim: int
    dim: int
    discretization: Literal["IM", "IMEX"]
    damping: bool
    r_min: float
    theta_max: float


class LinOSSSequenceMixer(eqx.Module):
    """LinOSS sequence mixer layer."""

    cfg: LinOSSSequenceMixerConfig
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: Literal["IM", "IMEX"]
    damping: bool

    def __init__(
        self,
        cfg: LinOSSSequenceMixerConfig,
        in_features: int,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS sequence mixer layer."""
        A_key, G_key, B_key, C_key, D_key, step_key, key = jr.split(key, 7)

        self.steps = normal(stddev=0.5)(step_key, (cfg.dim,))
        steps = nn.sigmoid(self.steps)

        if cfg.discretization == "IMEX" and cfg.damping:
            r_max = 1.0
            mags = jnp.sqrt(
                random.uniform(G_key, shape=(cfg.hidden_dim,)) * (r_max**2 - cfg.r_min**2)
                + cfg.r_min**2
            )
            self.G_diag = (1 - mags**2) / (steps * mags**2)
            G_diag = nn.relu(self.G_diag)

            theta = random.uniform(A_key, shape=(cfg.dim,)) * cfg.theta_max
            self.A_diag = map_theta_to_A(theta, G_diag, steps)
        else:
            self.G_diag = None
            self.A_diag = random.uniform(A_key, shape=(cfg.dim,))

        self.B = simple_uniform_init(
            B_key, shape=(cfg.dim, in_features, 2), std=1.0 / math.sqrt(in_features)
        )
        self.C = simple_uniform_init(
            C_key, shape=(in_features, cfg.hidden_dim, 2), std=1.0 / math.sqrt(cfg.hidden_dim)
        )
        self.D = normal(stddev=1.0)(D_key, (in_features,))

        self.discretization = cfg.discretization
        self.damping = cfg.damping

    def __call__(self, input_sequence: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LinOSS sequence mixer layer."""
        steps = nn.sigmoid(self.steps)

        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        if self.discretization == "IM":
            if self.damping:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(
                        self.discretization, self.damping
                    )
                )
            else:
                A_diag = nn.relu(self.A_diag)
                ys = apply_linoss_im(A_diag, B_complex, input_sequence, steps)
        elif self.discretization == "IMEX":
            if self.damping:
                G_diag = nn.relu(self.G_diag)
                A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
                A_boundary_high = (
                    2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)
                ) / steps**2
                A_diag = (
                    A_boundary_low
                    + nn.relu(self.A_diag - A_boundary_low)
                    - nn.relu(self.A_diag - A_boundary_high)
                )
                ys = apply_damped_linoss_imex(A_diag, G_diag, B_complex, input_sequence, steps)
            else:
                A_diag = nn.relu(self.A_diag)
                ys = apply_linoss_imex(A_diag, B_complex, input_sequence, steps)
        else:
            raise NotImplementedError(f"Discretization {self.discretization} not implemented")

        # Apply SequenceMixer Output Operations Cx + Du
        Cy = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        xs = Cy + Du

        return xs
