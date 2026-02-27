"""This module contains the base class for all channel mixers in Discretax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from discretax.utils.config_mixin import PartialModule


class AbstractChannelMixer(eqx.Module, ABC, PartialModule):
    """Abstract base class for all channel mixers.

    This class defines the interface for all channel mixers.

    Args:
        in_features: Input dimensionality.
        key: JAX random key for initialization.
        out_features: Optional output dimensionality. If None, defaults to in_features.
        *args: Additional arguments for the channel mixer.
        **kwargs: Additional keyword arguments for the channel mixer.
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        out_features: int | None = None,
        **kwargs,
    ):
        """Initialize the channel mixer."""

    # TODO: right now we are not using this lambda. But we should! Also return is_inexact_array.
    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for channel mixer parameters.

        Returns:
            A lambda function that filters the channel mixer parameters.
        """
        return lambda x: eqx.is_inexact_array(x)

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Forward pass of the channel mixer.

        Args:
            x: The input tensor to the channel mixer.

        Returns:
            The output of the channel mixer.
        """
