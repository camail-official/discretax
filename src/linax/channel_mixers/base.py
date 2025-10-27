"""This module contains the base class for all channel mixers in Linax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class ChannelMixerConfig(ABC):
    """Configuration for channel mixers.

    Attributes:
        None by default. Concrete configs define their own fields.
    """

    @abstractmethod
    def build(self, in_features: int, out_features: int | None, key: PRNGKeyArray) -> ChannelMixer:
        """Build channel mixer from config.

        Args:
            in_features: Input dimensionality.
            out_features: Optional output dimensionality. If None, defaults to in_features.
            key: JAX random key for initialization.

        Returns:
            The channel mixer instance.
        """


class ChannelMixer[ConfigType: ChannelMixerConfig](eqx.Module, ABC):
    """Abstract base class for all channel mixers.

    This class defines the interface for all channel mixers.

    Args:
        in_features: Input dimensionality.
        cfg: Configuration for the channel mixer.
        key: JAX random key for initialization.
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        *,
        out_features: int | None = None,
    ):
        """Initialize the channel mixer."""

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for channel mixer parameters.

        Returns:
            A lambda function that filters the channel mixer parameters.
        """
        return lambda _: True

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Forward pass of the channel mixer.

        Args:
            x: The input tensor to the channel mixer.

        Returns:
            The output of the channel mixer.
        """
