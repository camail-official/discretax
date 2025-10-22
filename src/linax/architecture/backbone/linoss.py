"""LinOSS encoder."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.backbone.base import Backbone, BackboneConfig
from linax.architecture.channel_mixers.glu import GLU
from linax.architecture.sequence_mixers.base import SequenceMixer, SequenceMixerConfig


class LinOSSBlock(eqx.Module):
    """A single block in the LinOSS Encoder.

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
        in_features: int,
        drop_rate: float,
        sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS Encoder Block.

        Args:
            in_features:
              Dimensionality of the hidden representations.
            drop_rate:
              Dropout rate for the GLU.
            sequence_mixer:
              The sequence mixer instance for this block.
            key:
              JAX random key for initialization of layers.
        """
        # TODO: make this a BatchNorm (I think this is what the original implementation does)
        self.norm = eqx.nn.LayerNorm(
            shape=in_features,
        )

        self.sequence_mixer = sequence_mixer

        self.glu = GLU(in_features, in_features, key=key)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Apply the LinOSS Encoder Block to the input sequence.

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


@dataclass
class LinOSSBackboneConfig(BackboneConfig):
    """Configuration for the LinOSS Backbone.

    This configuration class defines the hyperparameters and settings for the LinOSS backbone.
    It includes options for the model's architecture, training parameters, and behavior.

    Attributes:
        name:
          Name of the backbone.
        hidden_dim:
          Dimensionality of the hidden representations.
        num_blocks:
          Number of encoder blocks in the model.
        dropout_rate:
          Dropout rate for the GLU.
        classification:
          Whether the model is a classification model.
        sequence_mixer_config:
          Configuration for the sequence mixer.
    """

    name: str = "linoss_backbone"
    hidden_dim: int = 64
    num_blocks: int = 4
    dropout_rate: float = 0.1
    classification: bool = True
    sequence_mixer_config: SequenceMixerConfig | None = None


class LinOSSBackbone[ConfigType: LinOSSBackboneConfig](Backbone):
    """LinOSS Backbone.

    This model implements a sequence of encoder blocks that process input sequences.
    It includes linear encoders and decoders for dimensionality reduction and reconstruction.

    Attributes:
        linear_encoder:
          Linear encoder for dimensionality reduction.
        linear_decoder:
          Linear decoder for reconstruction.
        blocks:
          List of encoder blocks for sequence processing.
        hidden_dim:
          Dimensionality of the hidden representations.
        classification:
          Whether the model is a classification model.
        _out_features:
          Cached output features.
    """

    linear_encoder: eqx.nn.Linear
    linear_decoder: eqx.nn.Linear
    blocks: list[LinOSSBlock]
    hidden_dim: int
    classification: bool
    _out_features: int

    def __init__(
        self,
        cfg: ConfigType,
        in_features: int,
        key: PRNGKeyArray,
        sequence_mixers: list[SequenceMixer],
        out_features: int,
    ):
        """Initialize the LinOSS Backbone.

        Args:
            cfg:
              Configuration for the backbone.
            in_features:
              Dimensionality of the input features.
            key:
              JAX random key for initialization of layers.
            sequence_mixers:
              List of sequence mixer instances, one per block.
            out_features:
              Dimensionality of the output features.

        Raises:
            ValueError:
              If the number of sequence mixers does not match the number of blocks.
        """
        if len(sequence_mixers) != cfg.num_blocks:
            raise ValueError(
                f"Number of sequence_mixers ({len(sequence_mixers)}) must match "
                f"num_blocks ({cfg.num_blocks})"
            )

        self.hidden_dim = cfg.hidden_dim
        self.classification = cfg.classification
        self._out_features = in_features if out_features is None else out_features

        key, linear_encoder_key, linear_decoder_key, *block_keys = jr.split(
            key, cfg.num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(
            in_features,
            cfg.hidden_dim,
            key=linear_encoder_key,
            use_bias=False,
        )
        self.linear_decoder = eqx.nn.Linear(
            cfg.hidden_dim,
            self._out_features,
            key=linear_decoder_key,
            use_bias=False,
        )

        # Create blocks with pre-instantiated sequence mixers
        self.blocks = [
            LinOSSBlock(
                in_features=cfg.hidden_dim,
                drop_rate=cfg.dropout_rate,
                sequence_mixer=mixer,
                key=b_key,
            )
            for mixer, b_key in zip(sequence_mixers, block_keys)
        ]

    @property
    def out_features(self) -> int:
        """Output dimensionality of the model."""
        return self._out_features

    def __call__(self, x, state, key):
        """Forward pass of the LinOSS Backbone.

        The forward pass applies the linear encoder, a sequence of encoder blocks,
        and the linear decoder. The output is either a classification logits or a
        regression output, depending on the model configuration.

        Args:
            x:
              Input tensor of shape (timesteps, in_features).
            state:
              Current state for stateful normalization layers.
            key:
              JAX random key for dropout operations.
        """
        dropout_keys = jr.split(key, len(self.blocks))
        y = jax.vmap(self.linear_encoder)(x)
        for i, (block, d_key) in enumerate(zip(self.blocks, dropout_keys)):
            y, state = block(y, state, key=d_key)

        x = jax.vmap(self.linear_decoder)(y)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.log_softmax(x, axis=-1)
        else:
            x = jnp.mean(x, axis=0)

        return x, state
