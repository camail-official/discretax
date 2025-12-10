"""Standard block. This is the standard block used in recent papers like S5, LRU and LinOSS."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.base import AbstractBlock
from discretax.channel_mixers.base import AbstractChannelMixer
from discretax.sequence_mixers.base import AbstractSequenceMixer
from discretax.utils.config_mixin import Resolvable


class StandardBlock(AbstractBlock):
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
    """

    norm: eqx.nn.BatchNorm
    sequence_mixer: AbstractSequenceMixer
    channel_mixer: AbstractChannelMixer
    drop: eqx.nn.Dropout
    prenorm: bool

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *,
        sequence_mixer: Resolvable[AbstractSequenceMixer],
        channel_mixer: Resolvable[AbstractChannelMixer],
        drop_rate: float = 0.1,
        prenorm: bool = True,
        **kwargs,
    ):
        """Initialize the Standard block.

        Args:
            in_features: input features.
            key: JAX random key for initialization of layers.
            sequence_mixer: the sequence mixer instance for this block.
            channel_mixer: the channel mixer instance for this block.
            drop_rate: dropout rate for the channel mixer.
            prenorm: whether to apply the normalization at the beginning or the end of the block.
            **kwargs: Additional keyword arguments for the block.
        """
        self.norm = eqx.nn.BatchNorm(
            input_size=in_features, axis_name="batch", channelwise_affine=False, mode="ema"
        )

        # Build the sequence mixer and channel mixer from the config or an instance.
        self.sequence_mixer = sequence_mixer.resolve(in_features=in_features, key=key, **kwargs)
        self.channel_mixer = channel_mixer.resolve(in_features=in_features, key=key, **kwargs)

        self.drop = eqx.nn.Dropout(p=drop_rate)
        self.prenorm = prenorm

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
