"""LinOSS encoder."""

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.layers.glu import GLU
from linax.models.base import AbstractModel, AbstractModelConfig
from linax.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)


class LinossEncoderBlock(eqx.Module):
    """A single block in the LinOSS Encoder.

    This block implements a sequence mixer, normalization layers, and a GLU-based MLP.

    Attributes:
        norm:
          RMSNorm layer applied before the sequence mixer.
        sequence_mixer:
          The sequence mixing mechanism for sequence processing.
        glu:
          GLU-based feed-forward network.
        drop:
          Dropout layer applied after the GLU.
    """

    norm: eqx.nn.BatchNorm
    sequence_mixer: LinOSSSequenceMixer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        hidden_dim: int,
        drop_rate: float,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS Encoder Block.

        Args:
            hidden_dim:
              Dimensionality of the hidden representations.
            drop_rate:
              Dropout rate for the GLU.
            key:
              JAX random key for initialization of layers.
        """

        sequence_mixer_key, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim,
            axis_name="batch",
            channelwise_affine=False,
            mode="ema",
        )
        self.sequence_mixer = LinOSSSequenceMixer(
            cfg=LinOSSSequenceMixerConfig(
                hidden_dim=hidden_dim,
                dim=hidden_dim,
                discretization="IMEX",
                damping=True,
                r_min=0.9,
                theta_max=jnp.pi,
            ),
            in_features=hidden_dim,
            key=sequence_mixer_key,
        )
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
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
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


@dataclass
class LinossModelConfig(AbstractModelConfig):
    """Configuration for the LinOSS Model.

    This configuration class defines the hyperparameters and settings for the LinOSS model.
    It includes options for the model's architecture, training parameters, and behavior.

    Attributes:
        hidden_dim:
          Dimensionality of the hidden representations.
        num_blocks:
          Number of encoder blocks in the model.
        dropout_rate:
          Dropout rate for the GLU.
        name:
          Name of the model.
        classification:
          Whether the model is a classification model.
        sequence_mixer_config:
          Configuration for the sequence mixer.
    """

    hidden_dim: int = 64
    num_blocks: int = 4
    dropout_rate: float = 0.1
    name: Literal["linoss"] = "linoss"
    classification: bool = True
    # TODO: Add the sequence mixer config here (there should be a sensible default)


class LinossModel(AbstractModel[LinossModelConfig]):
    """LinOSS Model.

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
    """

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        out_features: int | None = None,
        cfg: LinossModelConfig = LinossModelConfig(),
    ):
        """Initialize the LinOSS Model.

        Args:
            in_features:
              Dimensionality of the input features.
            key:
              JAX random key for initialization of layers.
            out_features:
              Dimensionality of the output features.
            cfg:
              Configuration for the model.
        """
        self.hidden_dim = cfg.hidden_dim
        self.classification = cfg.classification
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
            in_features if out_features is None else out_features,
            key=linear_decoder_key,
            use_bias=False,
        )

        key, *block_keys = jr.split(key, cfg.num_blocks + 1)
        self.blocks = [
            LinossEncoderBlock(
                cfg.hidden_dim,
                key=b_key,
                drop_rate=cfg.dropout_rate,
            )
            for b_key in block_keys
        ]

    @property
    def out_features(self) -> int:
        """Output dimensionality of the model."""
        return self.cfg.hidden_dim

    def __call__(self, x, state, key):
        """Forward pass of the LinOSS Model.

        The forward pass applies the linear encoder, a sequence of encoder blocks, and the linear decoder.
        The output is either a classification logits or a regression output, depending on the model configuration.

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
