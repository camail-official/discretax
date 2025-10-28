"""S5 model configuration."""

from dataclasses import dataclass, field

from linax.blocks.standard import StandardBlockConfig
from linax.channel_mixers.glu import GLUConfig
from linax.encoder.base import EncoderConfig
from linax.heads.base import HeadConfig
from linax.models.ssm import SSMConfig
from linax.sequence_mixers.s5 import S5SequenceMixerConfig


@dataclass(frozen=True)
class S5Config(SSMConfig):
    """Configuration for S5 models.

    This is a modular configuration that allows building an S5 model with different components.

    Attributes:
        num_blocks: Number of S5 blocks to stack.
        encoder_config: Configuration for the encoder.
        head_config: Configuration for the output head.
        sequence_mixer_config: Optional S5 sequence mixer config that will be replicated
            for each block. If not provided, defaults to S5SequenceMixerConfig().
        block_config: Optional S5 block config that will be replicated for each block.
            If not provided, defaults to StandardBlockConfig.

    Example:
        ```python
        # With explicit configs
        config = S5Config(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            sequence_mixer_config=S5SequenceMixerConfig(
                state_dim=64,
                ssm_blocks=1,
                conj_sym=True,
                clip_eigs=True,
                discretization="zoh",
            ),
            block_config=StandardBlockConfig(drop_rate=0.05),
            head_config=ClassificationHeadConfig(out_features=10),
        )

        # With defaults (simpler)
        config = S5Config(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            head_config=ClassificationHeadConfig(out_features=10),
        )
        model = config.build(key=key)
        ```

    Reference:
        S5: https://openreview.net/pdf?id=Ai8Hw3AXqks
    """

    num_blocks: int
    encoder_config: EncoderConfig
    head_config: HeadConfig
    sequence_mixer_config: S5SequenceMixerConfig = field(default_factory=S5SequenceMixerConfig)
    block_config: StandardBlockConfig = field(default_factory=StandardBlockConfig)
    channel_mixer_config: GLUConfig = field(default_factory=GLUConfig)

    # These will be auto-populated from the single configs
    sequence_mixer_configs: list[S5SequenceMixerConfig] = field(init=False)
    block_configs: list[StandardBlockConfig] = field(init=False)
    channel_mixer_configs: list[GLUConfig] = field(init=False)

    def __post_init__(self):
        """Replicates configs for each block and validates."""
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(
            self, "sequence_mixer_configs", [self.sequence_mixer_config] * self.num_blocks
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

    cfg = S5Config(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    print("S5 Config:")
    print(f"  Num blocks: {cfg.num_blocks}")
    print(f"  Auto state_dim: {cfg.sequence_mixer_config.state_dim}")
    print(f"  Auto ssm_blocks: {cfg.sequence_mixer_config.ssm_blocks}")
    print(f"  Auto conj_sym: {cfg.sequence_mixer_config.conj_sym}")
    print(f"  Auto clip_eigs: {cfg.sequence_mixer_config.clip_eigs}")
    print(f"  Auto discretization: {cfg.sequence_mixer_config.discretization}")
    print(f"  Auto drop_rate: {cfg.block_config.drop_rate}\n")

    s5 = cfg.build(key=jr.PRNGKey(0))
    print(s5)
