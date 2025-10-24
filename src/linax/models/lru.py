"""LRU model configuration."""

from dataclasses import dataclass, field

from linax.blocks.lru import LRUBlockConfig
from linax.encoder.base import EncoderConfig
from linax.heads.base import HeadConfig
from linax.models.ssm import SSMConfig
from linax.sequence_mixers.lru import LRUSequenceMixerConfig


@dataclass(frozen=True)
class LRUConfig(SSMConfig):
    """Configuration for LRU models.

    This is a modular configuration that allows building an LRU model with different components.

    Attributes:
        num_blocks: Number of LRU blocks to stack.
        encoder_config: Configuration for the encoder.
        head_config: Configuration for the output head.
        sequence_mixer_config: Optional LRU sequence mixer config that will be replicated
            for each block. If not provided, defaults to LRUSequenceMixerConfig().
        block_config: Optional LRU block config that will be replicated for each block.
            If not provided, defaults to LRUBlockConfig.

    Example:
        ```python
        # With explicit configs
        config = LRUConfig(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            sequence_mixer_config=LRUSequenceMixerConfig(
                state_dim=64,
                r_min=0.0,
                r_max=1.0,
                max_phase=6.28,
            ),
            block_config=LRUBlockConfig(drop_rate=0.1),
            head_config=ClassificationHeadConfig(out_features=10),
        )

        # With defaults (simpler)
        config = LRUConfig(
            num_blocks=4,
            encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
            head_config=ClassificationHeadConfig(out_features=10),
        )
        model = config.build(key=key)
        ```

    Reference:
        LRU: https://arxiv.org/abs/2303.06349
    """

    num_blocks: int
    encoder_config: EncoderConfig
    head_config: HeadConfig
    sequence_mixer_config: LRUSequenceMixerConfig = field(default_factory=LRUSequenceMixerConfig)
    block_config: LRUBlockConfig = field(default_factory=LRUBlockConfig)

    # These will be auto-populated from the single configs
    sequence_mixer_configs: list[LRUSequenceMixerConfig] = field(init=False)
    block_configs: list[LRUBlockConfig] = field(init=False)

    def __post_init__(self):
        """Replicates configs for each block and validates."""
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(
            self, "sequence_mixer_configs", [self.sequence_mixer_config] * self.num_blocks
        )
        object.__setattr__(self, "block_configs", [self.block_config] * self.num_blocks)

        super().__post_init__()


if __name__ == "__main__":
    import jax.random as jr

    from linax.encoder import LinearEncoderConfig
    from linax.heads.classification import ClassificationHeadConfig

    cfg = LRUConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=784, out_features=64),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    print("LRU Config:")
    print(f"  Num blocks: {cfg.num_blocks}")
    print(f"  Auto state_dim: {cfg.sequence_mixer_config.state_dim}")
    print(f"  Auto r_min: {cfg.sequence_mixer_config.r_min}")
    print(f"  Auto r_max: {cfg.sequence_mixer_config.r_max}")
    print(f"  Auto max_phase: {cfg.sequence_mixer_config.max_phase}")
    print(f"  Auto drop_rate: {cfg.block_config.drop_rate}\n")

    lru = cfg.build(key=jr.PRNGKey(0))
    print(lru)
