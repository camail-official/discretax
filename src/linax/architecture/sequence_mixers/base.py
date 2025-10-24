"""This module contains the base class for all sequence mixers in Linax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class SequenceMixerConfig(ABC):
    """Configuration for sequence mixers.

    Attributes:
        state_dim:
          Dimensionality of the state space.
    """

    state_dim: int

    @abstractmethod
    def build(self, in_features: int, key: PRNGKeyArray) -> SequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features:
              Input dimensionality.
            key:
              JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """


class SequenceMixer[ConfigType: SequenceMixerConfig](eqx.Module, ABC):
    """Abstract base class for all sequence mixers.

    This class is used to define the interface for all sequence mixers.

    Args:
        in_features:
          Input dimensionality.
        cfg:
          Configuration for the sequence mixer.
        key:
          JAX random key for initialization.
        **kwargs:
          Additional keyword arguments for specific sequence mixer implementations.
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """Initialize the sequence mixer."""

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for sequence mixer parameters.

        Returns:
            A lambda function that filters the sequence mixer parameters.
        """
        return lambda _: True

    @abstractmethod
    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the sequence mixer.

        Args:
            x:
              The input sequence to the sequence mixer.
            key:
              The random key for the sequence mixer.

        Returns:
            The output of the sequence mixer.
        """
