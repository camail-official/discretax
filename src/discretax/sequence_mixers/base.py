"""This module contains the base class for all sequence mixers in Discretax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from discretax.utils.config_mixin import PartialModule


class AbstractSequenceMixer(eqx.Module, ABC, PartialModule):
    """Abstract base class for all sequence mixers.

    This class is used to define the interface for all sequence mixers.

    Args:
        in_features: Input dimensionality.
        key: JAX random key for initialization.
        *args: Additional arguments for the sequence mixer.
        **kwargs: Additional keyword arguments for the sequence mixer.
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ):
        """Initialize the sequence mixer."""
        raise NotImplementedError("Subclasses must implement __init__")

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for sequence mixer parameters.

        Returns:
            A lambda function that filters the sequence mixer parameters.
        """
        return lambda x: eqx.is_inexact_array(x)

    @abstractmethod
    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the sequence mixer.

        Args:
            x: The input sequence to the sequence mixer.
            key: The random key for the sequence mixer.

        Returns:
            The output of the sequence mixer.
        """
