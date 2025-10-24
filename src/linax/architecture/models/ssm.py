"""General SSM (State Space Model) implementation."""

from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.blocks.base import Block, BlockConfig
from linax.architecture.encoder.base import Encoder, EncoderConfig
from linax.architecture.heads.base import Head, HeadConfig
from linax.architecture.models.base import AbstractModel, ModelConfig
from linax.architecture.sequence_mixers.base import SequenceMixerConfig


@dataclass
class SSMConfig(ModelConfig):
    """Configuration for SSM models.

    Attributes:
        hidden_dim:
          Dimensionality of the hidden state.
        encoder_config:
          Configuration for the encoder.
        sequence_mixer_configs:
          Configuration for the sequence mixers.
        block_configs:
          Configuration for the blocks.
        head_config:
          Configuration for the head.

    Raises:
        ValueError: If the number of sequence mixers and blocks is not the same.
    """

    hidden_dim: int
    encoder_config: EncoderConfig
    sequence_mixer_configs: list[SequenceMixerConfig]
    block_configs: list[BlockConfig]
    head_config: HeadConfig

    def __post_init__(self):
        """Validate config."""
        if len(self.sequence_mixer_configs) != len(self.block_configs):
            raise ValueError("sequence_mixer_configs and block_configs must have same length")


class SSM[ConfigType: SSMConfig](AbstractModel):
    """General SSM model.

    Args:
        cfg:
          Configuration for the SSM model.
        key:
          JAX random key for initialization.

    Attributes:
        encoder:
          Encoder instance.
        blocks:
          List of blocks.
        head:
          Head instance.
    """

    encoder: Encoder
    blocks: list[Block]
    head: Head

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        num_blocks = len(cfg.block_configs)
        keys = jr.split(key, 2 * num_blocks + 2)

        self.encoder = cfg.encoder_config.build(out_features=cfg.hidden_dim, key=keys[0])

        sequence_mixers = [
            mixer_cfg.build(in_features=cfg.hidden_dim, key=keys[1 + i])
            for i, mixer_cfg in enumerate(cfg.sequence_mixer_configs)
        ]

        self.blocks = [
            block_cfg.build(
                in_features=cfg.hidden_dim, sequence_mixer=mixer, key=keys[1 + num_blocks + i]
            )
            for i, (block_cfg, mixer) in enumerate(zip(cfg.block_configs, sequence_mixers))
        ]

        self.head = cfg.head_config.build(in_features=cfg.hidden_dim, key=keys[-1])

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the SSM model.

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
        # Prepare the keys
        block_keys = jr.split(key, len(self.blocks))

        # Encode the input
        x, state = self.encoder(x, state)

        # Apply the blocks
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)

        # Apply the head
        x, state = self.head(x, state)
        return x, state
