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
    """Low-level configuration for State Space Models.

    This is a component-based configuration that provides fine-grained control over
    the SSM architecture. You must explicitly specify configurations for each
    component: encoder, sequence mixers, blocks, and head.

    Use this when:
    - Building custom SSM architectures
    - Mixing different component types
    - Needing full control over each component's configuration

    For pre-configured architectures (e.g., LinOSS), use high-level configs like
    `LinOSSConfig` which automatically compose the appropriate components.

    Attributes:
        hidden_dim:
          Dimensionality of the hidden state throughout the model.
        encoder_config:
          Configuration for the encoder that processes input data.
        sequence_mixer_configs:
          List of configurations for sequence mixers, one per block.
        block_configs:
          List of configurations for blocks, one per sequence mixer.
        head_config:
          Configuration for the output head.

    Raises:
        ValueError: If the number of sequence_mixer_configs and block_configs differ.

    Example:
        ```python
        config = SSMConfig(
            hidden_dim=128,
            encoder_config=LinearEncoderConfig(in_features=784),
            sequence_mixer_configs=[LinOSSSequenceMixerConfig(state_dim=128)] * 4,
            block_configs=[LinOSSBlockConfig(drop_rate=0.1)] * 4,
            head_config=ClassificationHeadConfig(out_features=10),
        )
        model = SSM(cfg=config, key=key)
        ```
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
    """General State Space Model (SSM) implementation.

    This is a flexible, composable SSM architecture that can be configured with
    different encoders, sequence mixers, blocks, and heads. It serves as the
    base implementation for all SSM variants in linax.

    The model applies components in the following order:
    1. Encoder: Transforms input to hidden dimension
    2. Blocks: Stack of (sequence mixer + channel mixer) layers
    3. Head: Produces final output (classification, regression, etc.)

    Args:
        cfg:
          Low-level configuration specifying all components (see `SSMConfig`).
        key:
          JAX random key for parameter initialization.

    Attributes:
        encoder:
          The encoder instance that processes raw inputs.
        blocks:
          List of block instances, each containing a sequence mixer and channel mixer.
        head:
          The output head instance that produces final predictions.
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
