"""Block base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.sequence_mixers.base import SequenceMixer


@dataclass(frozen=True)
class BlockConfig(ABC):
    """Configuration for blocks."""


class Block[ConfigType: BlockConfig](eqx.Module, ABC):
    """Abstract base class for all blocks."""

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the block.

        Args:
            in_features:
              Input features.
            cfg:
              Configuration for the block.
            sequence_mixer:
              The sequence mixer instance for this block.
            key:
              JAX random key for initialization.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the block.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.
            key:
              JAX random key for operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        pass
