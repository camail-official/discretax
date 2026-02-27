"""This module contains the base class for all blocks in Discretax."""

from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from discretax.channel_mixers.base import AbstractChannelMixer
from discretax.sequence_mixers.base import AbstractSequenceMixer
from discretax.utils.config_mixin import PartialModule


class AbstractBlock(eqx.nn.StatefulLayer, ABC, PartialModule):
    """Abstract base class for all blocks.

    Args:
        in_features: Input features.
        sequence_mixer: The sequence mixer instance for this block.
        channel_mixer: The channel mixer instance for this block.
        key: JAX random key for initialization.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).
    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        sequence_mixer: AbstractSequenceMixer,
        channel_mixer: AbstractChannelMixer,
        **kwargs,
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
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
