"""This module contains the base class for all blocks in Linax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.sequence_mixers.base import SequenceMixer


@dataclass(frozen=True)
class BlockConfig(ABC):
    """Configuration for blocks."""

    @abstractmethod
    def build(self, in_features: int, sequence_mixer: SequenceMixer, key: PRNGKeyArray) -> Block:
        """Build block from config.

        Args:
            in_features:
              Input features.
            sequence_mixer:
              The sequence mixer instance for this block.
            key:
              JAX random key for initialization.

        Returns:
            The block instance.
        """


class Block[ConfigType: BlockConfig](eqx.Module, ABC):
    """Abstract base class for all blocks.

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

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the block."""

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
