"""LinOSS block."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.blocks.base import Block, BlockConfig
from linax.architecture.channel_mixers.glu import GLU
from linax.architecture.sequence_mixers.base import SequenceMixer


@dataclass
class LinOSSBlockConfig(BlockConfig):
    """Configuration for the LinOSS block.

    Attributes:
        name:
          Name of the block.
        in_features:
          Dimensionality of the input features.
        drop_rate:
          Dropout rate for the GLU.

    """

    name: str = "linoss_block"
    in_features: int = 64

    drop_rate: float = 0.1


class LinOSSBlock[ConfigType: LinOSSBlockConfig](Block):
    """A single block in the LinOSS backbone.

    This block implements a sequence mixer, normalization layers, and a GLU-based MLP.

    Attributes:
        norm:
          LayerNorm layer applied after the sequence mixer.
        sequence_mixer:
          The sequence mixing mechanism for sequence processing.
        glu:
          GLU-based feed-forward network.
        drop:
          Dropout layer applied after the GLU.
    """

    norm: eqx.nn.LayerNorm
    sequence_mixer: SequenceMixer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        cfg: ConfigType,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS block.

        Args:
            cfg:
              Configuration for the LinOSS block.
            in_features:
              Dimensionality of the hidden representations.
            sequence_mixer:
              The sequence mixer instance for this block.
            key:
              JAX random key for initialization of layers.
        """
        # TODO: make this a BatchNorm (I think this is what the original implementation does)
        self.norm = eqx.nn.LayerNorm(
            shape=cfg.in_features,
        )

        self.sequence_mixer = sequence_mixer

        self.glu = GLU(cfg.in_features, cfg.in_features, key=key)
        self.drop = eqx.nn.Dropout(p=cfg.drop_rate)

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Apply the LinOSS block to the input sequence.

        Args:
            x:
              Input tensor of shape (timesteps, hidden_dim).
            state:
              Current state for stateful normalization layers.
            key:
              JAX random key for dropout operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        key, dropkey1, dropkey2 = jr.split(key, 3)
        skip = x
        x = self.sequence_mixer(x, key)
        x, state = jax.vmap(self.norm)(x, state)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state
