"""LRU block.

Adapted from https://github.com/tk-rusch/linoss/blob/main/models/LRU.py
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.blocks.base import Block, BlockConfig
from linax.channel_mixers.glu import GLU
from linax.sequence_mixers.base import SequenceMixer
from linax.utils import count_params


@dataclass(frozen=True)
class LRUBlockConfig(BlockConfig):
    """Configuration for the LRU block.

    Attributes:
        drop_rate: Dropout rate for the GLU.
    """

    drop_rate: float = 0.1

    def build(
        self, in_features: int, sequence_mixer: SequenceMixer, key: PRNGKeyArray
    ) -> LRUBlock:
        """Build block from config.

        Args:
            in_features: Input features.
            sequence_mixer: The sequence mixer instance for this block.
            key: JAX random key for initialization of layers.

        Returns:
            The LRU block instance.
        """
        return LRUBlock(in_features=in_features, cfg=self, sequence_mixer=sequence_mixer, key=key)


class LRUBlock[ConfigType: LRUBlockConfig](Block):
    """A single block in the LRU backbone.

    This block implements a sequence mixer, normalization layers, and a GLU-based MLP.

    Attributes:
        norm: LayerNorm layer applied after the sequence mixer.
        sequence_mixer: The sequence mixing mechanism for sequence processing.
        mlp: GLU-based feed-forward network.
        drop: Dropout layer applied after the GLU.
    """

    norm: eqx.nn.LayerNorm
    sequence_mixer: SequenceMixer
    # TODO: allow for a general MLP here
    mlp: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the LRU block.

        Args:
            in_features: Input features.
            cfg: Configuration for the LRU block.
            sequence_mixer: The sequence mixer instance for this block.
            key: JAX random key for initialization of layers.
        """
        # TODO: this should be a BatchNorm
        self.norm = eqx.nn.LayerNorm(shape=in_features)

        self.sequence_mixer = sequence_mixer

        self.mlp = GLU(input_dim=in_features, output_dim=in_features, key=key)
        self.drop = eqx.nn.Dropout(p=cfg.drop_rate)

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Apply the LRU block to the input sequence.

        Args:
            x: Input tensor of shape (timesteps, hidden_dim).
            state: Current state for stateful normalization layers.
            key: JAX random key for dropout operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        lrukey, dropkey1, dropkey2 = jr.split(key, 3)

        # TODO This should be a batchnorm. how do we do batchnorm
        # if the batch dimension is vmapped?
        skip = x
        x, state = jax.vmap(self.norm)(x, state)
        x = self.sequence_mixer(x, lrukey)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.mlp)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state

    def __repr__(self) -> str:
        """Return a string representation of the LRU block.

        Returns:
            Compact summary of block configuration and components.
        """
        dropout_rate = self.drop.p
        norm_type = type(self.norm).__name__

        # Get LRU-specific config if available
        # TODO philipp @francesco should every sequence mixer have its own repr
        # which is then called by the block? Would probably make more sense...
        # TODO: should do something like self.sequence_mixer.__repr__()
        r_min = getattr(self.sequence_mixer, "r_min", "N/A")
        r_max = getattr(self.sequence_mixer, "r_max", "N/A")

        # Get MLP representation
        mlp_repr = repr(self.mlp)

        params = count_params(self)

        return (
            f"{params:,} params | {norm_type} | r:[{r_min},{r_max}] | "
            f"{mlp_repr} | drop:{dropout_rate:.2f}"
        )
