"""Sequence mixer base class."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.base import AbstractConfig


@dataclass
class SequenceMixerConfig(AbstractConfig):
    """Configuration for sequence mixers with no additional attributes."""

    name: str = "sequence_mixer"


class SequenceMixer[ConfigType: SequenceMixerConfig](eqx.Module):
    """Abstract base class for all sequence mixers.

    This class is used to define the interface for all sequence mixers.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        in_features: int,
        key: PRNGKeyArray,
        **kwargs,
    ):
        pass

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for sequence mixer parameters.

        Returns:
            A lambda function that filters the sequence mixer parameters.
        """
        return lambda _: True

    @abstractmethod
    def __call__(self, input_sequence: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the sequence mixer.

        Args:
            input_sequence: The input sequence to the sequence mixer.
            key: The random key for the sequence mixer.

        Returns:
            The output of the sequence mixer.
        """
        pass
