"""This module contains the base class for all heads in Discretax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from discretax.utils.config_mixin import PartialLoaderMixin


class AbstractHead(eqx.nn.StatefulLayer, ABC, PartialLoaderMixin):
    """Abstract base class for all heads.

    This is the base class for all heads in Discretax.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        key: JAX random key for initialization.
        *args: Additional arguments for the head.
        **kwargs: Additional keyword arguments for the head.
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ):
        """Initialize the head."""
        raise NotImplementedError("Subclasses must implement __init__")

    @abstractmethod
    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the head.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: Optional JAX random key (unused by heads, for Sequential compatibility).

        Returns:
            Tuple containing the output tensor and updated state.
        """

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for head parameters."""
        return lambda x: eqx.is_inexact_array(x)
