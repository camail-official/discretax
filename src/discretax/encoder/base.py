"""This module contains the base class for all encoders in Discretax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from discretax.utils.config_mixin import Cfg, PartialLoaderMixin


class Encoder(eqx.nn.StatefulLayer, ABC, PartialLoaderMixin):
    """Abstract base class for all encoders.

    This is the base class for all encoders.

    Args:
        key: JAX random key for initialization.
        *args: Additional arguments for the encoder.
        **kwargs: Additional keyword arguments for the encoder.
    """

    @abstractmethod
    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        out_features: Cfg[int],
        **kwargs,
    ):
        """Initialize the encoder."""

    @abstractmethod
    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the encoder.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: Optional JAX random key (unused by encoders, for Sequential compatibility).

        Returns:
            Tuple containing the output tensor and updated state.
        """

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for encoder parameters."""
        return lambda x: eqx.is_inexact_array(x)
