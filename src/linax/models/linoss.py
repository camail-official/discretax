"""LinOSS model configuration."""

from dataclasses import dataclass, field

from linax.blocks.standard import StandardBlockConfig
from linax.channel_mixers.glu import GLUConfig
from linax.encoder.base import EncoderConfig
from linax.heads.base import HeadConfig
from linax.models.ssm import SSMConfig
from linax.sequence_mixers.linoss import LinOSSSequenceMixerConfig


@dataclass(frozen=True)
class LinOSSConfig(SSMConfig):
    """Configuration for LinOSS models.

    This is a modular configuration that allows building a LinOSS model with different components.

    Attributes:
        num_blocks: Number of LinOSS blocks to stack.
        encoder_config: Configuration for the encoder (contains in_features and out_features).
        head_config: Configuration for the output head (contains out_features).
        sequence_mixer_config: Optional linoss sequence mixer config that will be replicated
            for each block. If not provided, defaults to LinOSSSequenceMixerConfig().
        block_config: Optional linoss block config that will be replicated for each block.
            If not provided, defaults to StandardBlockConfig.

    Example:
        ```python
        # With explicit configs
        config = LinOSSConfig(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=64),
            block_config=StandardBlockConfig(drop_rate=0.1),
            head_config=ClassificationHeadConfig(out_features=10),
        )

        # With defaults (simpler)
        config = LinOSSConfig(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            head_config=ClassificationHeadConfig(out_features=10),
        )
        model = config.build(key=key)
        ```

    Reference:
        LinOSS: https://openreview.net/pdf?id=GRMfXcAAFh
    """

    num_blocks: int
    encoder_config: EncoderConfig
    head_config: HeadConfig
    sequence_mixer_config: LinOSSSequenceMixerConfig = field(
        default_factory=LinOSSSequenceMixerConfig
    )
    block_config: StandardBlockConfig = field(default_factory=StandardBlockConfig)
    channel_mixer_config: GLUConfig = field(default_factory=GLUConfig)

    # These will be auto-populated from the single configs
    sequence_mixer_configs: list[LinOSSSequenceMixerConfig] = field(init=False)
    block_configs: list[StandardBlockConfig] = field(init=False)
    channel_mixer_configs: list[GLUConfig] = field(init=False)

    def __post_init__(self):
        """Replicates configs for each block and validates."""
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(
            self,
            "sequence_mixer_configs",
            [self.sequence_mixer_config] * self.num_blocks,
        )
        object.__setattr__(self, "block_configs", [self.block_config] * self.num_blocks)
        object.__setattr__(
            self, "channel_mixer_configs", [self.channel_mixer_config] * self.num_blocks
        )

        super().__post_init__()


if __name__ == "__main__":
    import jax.random as jr

    from linax.encoder import LinearEncoderConfig
    from linax.heads.classification import ClassificationHeadConfig

    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    print(f"  Auto state_dim: {cfg.sequence_mixer_config.state_dim}")
    print(f"  Auto drop_rate: {cfg.block_config.drop_rate}\n")

    linoss = cfg.build(key=jr.PRNGKey(0))
    print(linoss)
