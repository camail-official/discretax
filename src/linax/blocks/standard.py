"""Standard block. This is the standard block used in recent papers like S5, LRU and LinOSS."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.blocks.base import Block, BlockConfig
from linax.channel_mixers.base import ChannelMixer
from linax.sequence_mixers.base import SequenceMixer


@dataclass(frozen=True)
class StandardBlockConfig(BlockConfig):
    """Configuration for the Standard block.

    Attributes:
        drop_rate: Dropout rate for the channel mixer.
        prenorm: Whether to apply the normalization at the beginning or the end of the block.
    """

    drop_rate: float = 0.1
    prenorm: bool = True

    def build(
        self,
        in_features: int,
        sequence_mixer: SequenceMixer,
        channel_mixer: ChannelMixer,
        key: PRNGKeyArray,
    ) -> StandardBlock:
        """Build block from config.

        Args:
            in_features: Input features.
            sequence_mixer: The sequence mixer instance for this block.
            channel_mixer: The channel mixer instance for this block.
            key: JAX random key for initialization of layers.

        Returns:
            The Standard block instance.
        """
        return StandardBlock(
            in_features=in_features,
            cfg=self,
            sequence_mixer=sequence_mixer,
            channel_mixer=channel_mixer,
            key=key,
        )


class StandardBlock[ConfigType: StandardBlockConfig](Block):
    """A single block in the Standard backbone.

    This block implements a sequence mixer, BatchNorm normalization, and a channel mixer.

    !!! warning
        This block uses BatchNorm for normalization. When training with vmap, ensure you
        name the batch axis as "batch" for compatibility. Example:

        ```python
        # Correct usage with axis naming
        jax.vmap(model, axis_name="batch")
        # or
        jax.vmap(model, in_axes=(0, None, 0), axis_name="batch")
        ```

        This ensures BatchNorm can properly compute batch statistics across the named axis.

    Attributes:
        norm: BatchNorm layer applied after the sequence mixer.
        sequence_mixer: The sequence mixing mechanism for sequence processing.
        channel_mixer: The channel mixing mechanism for channel processing.
        drop: Dropout layer applied after the channel mixer.
        prenorm: Whether to apply the normalization at the beginning or the end of the block.

    Args:
        in_features: Input features.
        cfg: Configuration for the Standard block.
        sequence_mixer: The sequence mixer instance for this block.
        channel_mixer: The channel mixer instance for this block.
        key: JAX random key for initialization of layers.
    """

    norm: eqx.nn.BatchNorm
    sequence_mixer: SequenceMixer
    channel_mixer: ChannelMixer
    drop: eqx.nn.Dropout
    prenorm: bool

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        channel_mixer: ChannelMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the Standard block."""
        self.norm = eqx.nn.BatchNorm(
            input_size=in_features, axis_name="batch", channelwise_affine=False, mode="ema"
        )

        self.sequence_mixer = sequence_mixer
        self.channel_mixer = channel_mixer
        self.drop = eqx.nn.Dropout(p=cfg.drop_rate)
        self.prenorm = cfg.prenorm

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Apply the Standard block to the input sequence.

        Args:
            x: Input tensor of shape (timesteps, hidden_dim).
            state: Current state for stateful normalization layers.
            key: JAX random key for dropout operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        key, dropkey1, dropkey2 = jr.split(key, 3)
        skip = x
        if self.prenorm:
            x, state = self.norm(x.T, state)
            x = x.T
        x = self.sequence_mixer(x, key)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.channel_mixer)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        if not self.prenorm:
            x, state = self.norm(x.T, state)
            x = x.T

        return x, state
