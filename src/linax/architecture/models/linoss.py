"""LinOSS model."""

from dataclasses import dataclass, field

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.blocks.linoss import LinOSSBlock, LinOSSBlockConfig
from linax.architecture.encoder import LinearEncoder, LinearEncoderConfig
from linax.architecture.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from linax.architecture.models.base import AbstractModel, ModelConfig
from linax.architecture.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)


@dataclass
class LinOSSConfig(ModelConfig):
    """Configuration for the LinOSS model.

    This configuration class defines hyperparameters for LinOSS.

    Attributes:
        name:
          Name of the model.
        hidden_dim:
          Dimensionality of the hidden representations.
        in_features:
          Dimensionality of the input features.
        out_features:
          Dimensionality of the output features. Default is None. If None, set to in_features.
        sequence_mixer_config:
          Configuration for the sequence mixer.
        backbone_config:
          Configuration for the backbone.
    """

    in_features: int = 784
    hidden_dim: int = 20
    out_features: int = 10
    num_blocks: int = 4

    # Component configs - created in __post_init__ to avoid circular dependencies
    encoder_config: LinearEncoderConfig = field(default=None, init=False)
    sequence_mixer_config: LinOSSSequenceMixerConfig = field(default=None, init=False)
    block_config: LinOSSBlockConfig = field(default=None, init=False)
    head_config: ClassificationHeadConfig = field(default=None, init=False)

    def __post_init__(self):
        """Create component configs with proper dimensions."""
        # Create component configs with correct dimensions
        self.encoder_config = LinearEncoderConfig(in_features=self.in_features)
        self.sequence_mixer_config = LinOSSSequenceMixerConfig(state_dim=self.hidden_dim)
        self.block_config = LinOSSBlockConfig()
        self.head_config = ClassificationHeadConfig(out_features=self.out_features)


class LinOSS[ConfigType: LinOSSConfig](AbstractModel):
    """LinOSS model combining sequence mixer and backbone.

    Attributes:
        encoder:
          The encoder for input processing.
        sequence_mixers:
          List of sequence mixer instances, one per block.
        blocks:
          List of blocks for sequence processing.
        head:
          The classification head.
    """

    encoder: LinearEncoder
    sequence_mixers: list[LinOSSSequenceMixer]
    blocks: list[LinOSSBlock]
    head: ClassificationHead

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the LinOSS model.

        Args:
            cfg:
              Configuration for the model.
            key:
              JAX random key for initialization.
        """
        # Split keys for encoder, sequence mixers, blocks, and head
        keys = jr.split(key, 2 * cfg.num_blocks + 2)
        encoder_key = keys[0]
        mixer_keys = keys[1 : 1 + cfg.num_blocks]
        block_keys = keys[1 + cfg.num_blocks : 1 + 2 * cfg.num_blocks]
        head_key = keys[-1]

        # Create independent sequence mixers for each block
        self.sequence_mixers = [
            LinOSSSequenceMixer(
                in_features=cfg.hidden_dim,
                cfg=cfg.sequence_mixer_config,
                key=mixer_key,
            )
            for mixer_key in mixer_keys
        ]

        # Create blocks with pre-instantiated sequence mixers
        self.blocks = [
            LinOSSBlock(
                in_features=cfg.hidden_dim,
                cfg=cfg.block_config,
                sequence_mixer=mixer,
                key=b_key,
            )
            for mixer, b_key in zip(self.sequence_mixers, block_keys)
        ]

        self.encoder = LinearEncoder(
            out_features=cfg.hidden_dim,
            cfg=cfg.encoder_config,
            key=encoder_key,
        )

        self.head = ClassificationHead(
            in_features=cfg.hidden_dim,
            cfg=cfg.head_config,
            key=head_key,
        )

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the LinOSS model.

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
        # prepare the keys
        block_keys = jr.split(key, len(self.blocks))

        # encode the input
        x, state = self.encoder(x, state)

        # apply the blocks
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)

        # apply the head
        x, state = self.head(x, state)
        return x, state
