"""General SSM (State Space Model) implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.blocks.base import Block, BlockConfig
from linax.architecture.encoder.base import Encoder, EncoderConfig
from linax.architecture.heads.base import Head, HeadConfig
from linax.architecture.sequence_mixers.base import SequenceMixerConfig
from linax.utils import count_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSMConfig:
    """Low-level configuration for State Space Models.

    This is a fully modular, component-based configuration that provides fine-grained
    control over the SSM architecture. Each component config contains its own dimension
    parameters, making the configuration self-contained and composable.

    Use this when:
    - Building custom SSM architectures
    - Mixing different component types
    - Needing full control over each component's configuration

    For pre-configured architectures (e.g., LinOSS), use high-level configs like
    `LinOSSConfig` which automatically compose the appropriate components.

    Attributes:
        encoder_config:
          Configuration for the encoder that processes input data. Must specify
          in_features and out_features (hidden_dim).
        sequence_mixer_configs:
          List of configurations for sequence mixers, one per block. Must be compatible
          with encoder's out_features (hidden_dim).
        block_configs:
          List of configurations for blocks, one per sequence mixer.
        head_config:
          Configuration for the output head. Must specify out_features. The in_features
          will be automatically set to match the encoder's out_features.

    Raises:
        ValueError: If the number of sequence_mixer_configs and block_configs differ.

    Example:
        ```python
        config = SSMConfig(
            encoder_config=LinearEncoderConfig(in_features=784, out_features=128),
            sequence_mixer_configs=[LinOSSSequenceMixerConfig(state_dim=128)] * 4,
            block_configs=[LinOSSBlockConfig(drop_rate=0.1)] * 4,
            head_config=ClassificationHeadConfig(out_features=10),
        )
        model = config.build(key=key)
        ```
    """

    encoder_config: EncoderConfig
    sequence_mixer_configs: list[SequenceMixerConfig]
    block_configs: list[BlockConfig]
    head_config: HeadConfig

    def __post_init__(self):
        """Validate config."""
        # Check number of configs match
        if len(self.sequence_mixer_configs) != len(self.block_configs):
            raise ValueError("sequence_mixer_configs and block_configs must have same length")

    def build(self, key: PRNGKeyArray | None = None) -> SSM:
        """Build an SSM model from this configuration.

        Args:
            key:
              JAX random key for parameter initialization.

        Returns:
            Instantiated SSM model.

        Example:
            ```python
            config = SSMConfig(...)
            model = config.build(key=jr.PRNGKey(0))
            ```
        """
        if key is None:
            logger.warning("No key provided. Set automatically.")
            key = jr.PRNGKey(0)

        return SSM(cfg=self, key=key)


class SSM[ConfigType: SSMConfig](eqx.Module):
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

        # Build encoder from its config (config contains all dimension info)
        self.encoder = cfg.encoder_config.build(key=keys[0])

        # Get hidden_dim from encoder output
        hidden_dim = cfg.encoder_config.out_features

        sequence_mixers = [
            mixer_cfg.build(in_features=hidden_dim, key=keys[1 + i])
            for i, mixer_cfg in enumerate(cfg.sequence_mixer_configs)
        ]

        self.blocks = [
            block_cfg.build(
                in_features=hidden_dim,
                sequence_mixer=mixer,
                key=keys[1 + num_blocks + i],
            )
            for i, (block_cfg, mixer) in enumerate(zip(cfg.block_configs, sequence_mixers))
        ]

        # Build head from its config (pass in_features from encoder)
        self.head = cfg.head_config.build(in_features=hidden_dim, key=keys[-1])

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

    def __repr__(self) -> str:
        """Pretty print the model architecture.

        Aggregates representations from encoder, blocks, and head to provide
        a comprehensive model summary. Each component provides its own repr.

        Returns:
            Formatted string representation of the model.
        """
        # Get basic info
        num_blocks = len(self.blocks)
        encoder_type = type(self.encoder).__name__
        head_type = type(self.head).__name__

        # Count parameters
        total_params = count_params(self)
        encoder_params = count_params(self.encoder)
        head_params = count_params(self.head)
        blocks_params = sum(count_params(block) for block in self.blocks)

        # Fixed width for alignment
        width = 70

        def pad_line(text: str) -> str:
            """Pad line to fixed width."""
            return f"║ {text:<{width}} ║"

        # Get block type from first block
        block_type = type(self.blocks[0]).__name__ if self.blocks else "Block"

        lines = [
            "╔" + "═" * (width + 2) + "╗",
            pad_line(f"{type(self).__name__} Model Summary".center(width)),
            "╠" + "═" * (width + 2) + "╣",
            pad_line("Components:"),
            pad_line(f"  Encoder:  {encoder_type} ({encoder_params:,} params)"),
            pad_line(f"  Blocks:   {num_blocks}× {block_type} (total {blocks_params:,} params)"),
        ]

        # Add each block's representation
        for i, block in enumerate(self.blocks):
            block_repr = repr(block)
            lines.append(pad_line(f"    [{i}] {block_repr}"))

        lines.extend(
            [
                pad_line(f"  Head:     {head_type} ({head_params:,} params)"),
                "╠" + "═" * (width + 2) + "╣",
                pad_line(f"Total Parameters: {total_params:,}"),
                "╚" + "═" * (width + 2) + "╝",
            ]
        )

        return "\n".join(lines)
