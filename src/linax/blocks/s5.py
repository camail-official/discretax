"""S5 block.

Adapted from https://github.com/tk-rusch/linoss/blob/main/models/S5.py
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
class S5BlockConfig(BlockConfig):
    """Configuration for the S5 block.

    Attributes:
        drop_rate: Dropout rate for the GLU.
    """

    drop_rate: float = 0.05

    def build(self, in_features: int, sequence_mixer: SequenceMixer, key: PRNGKeyArray) -> S5Block:
        """Build block from config.

        Args:
            in_features: Input features.
            sequence_mixer: The sequence mixer instance for this block.
            key: JAX random key for initialization of layers.

        Returns:
            The S5 block instance.
        """
        return S5Block(in_features=in_features, cfg=self, sequence_mixer=sequence_mixer, key=key)


class S5Block[ConfigType: S5BlockConfig](Block):
    """A single block in the S5 backbone.

    This block implements a sequence mixer, normalization layers, and a GLU-based MLP.

    Attributes:
        norm: LayerNorm layer applied after the sequence mixer.
        sequence_mixer: The sequence mixing mechanism for sequence processing.
        mlp: GLU-based feed-forward network.
        drop: Dropout layer applied after the GLU.
    """

    norm: eqx.nn.LayerNorm
    sequence_mixer: SequenceMixer
    mlp: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the S5 block.

        Args:
            in_features: Input features.
            cfg: Configuration for the S5 block.
            sequence_mixer: The sequence mixer instance for this block.
            key: JAX random key for initialization of layers.
        """
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
        """Apply the S5 block to the input sequence.

        Args:
            x: Input tensor of shape (timesteps, hidden_dim).
            state: Current state for stateful normalization layers.
            key: JAX random key for dropout operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        s5key, dropkey1, dropkey2 = jr.split(key, 3)

        skip = x
        # TODO: Should be BatchNorm. Also,
        # all of these block seem to be the same?!
        x, state = jax.vmap(self.norm)(x, state)
        x = self.sequence_mixer(x, s5key)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.mlp)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state

    def __repr__(self) -> str:
        """Return a string representation of the S5 block.

        Returns:
            Compact summary of block configuration and components.
        """
        dropout_rate = self.drop.p
        norm_type = type(self.norm).__name__

        # Get S5-specific config if available
        discretization = getattr(self.sequence_mixer, "discretization", "N/A")
        conj_sym = "✓" if getattr(self.sequence_mixer, "conj_sym", False) else "✗"
        clip_eigs = "✓" if getattr(self.sequence_mixer, "clip_eigs", False) else "✗"

        # Get MLP representation
        mlp_repr = repr(self.mlp)

        params = count_params(self)

        return (
            f"{params:,} params | {norm_type} | {discretization} | "
            f"conj:{conj_sym} | clip:{clip_eigs} | {mlp_repr} | drop:{dropout_rate:.2f}"
        )
