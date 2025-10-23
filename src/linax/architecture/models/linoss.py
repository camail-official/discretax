"""LinOSS model."""

from dataclasses import dataclass

import jax.random as jr
from jaxtyping import PRNGKeyArray

from linax.architecture.blocks.linoss import LinOSSBlockConfig
from linax.architecture.encoder import LinearEncoderConfig
from linax.architecture.heads.classification import (
    ClassificationHeadConfig,
)
from linax.architecture.models.ssm import SSM, SSMConfig
from linax.architecture.sequence_mixers.linoss import (
    LinOSSSequenceMixerConfig,
)


@dataclass
class LinOSSConfig:
    """Configuration for LinOSS model."""

    in_features: int = 784
    hidden_dim: int = 20
    out_features: int = 10
    num_blocks: int = 4
    drop_rate: float = 0.1

    def build(self) -> SSMConfig:
        """Build SSM config."""
        return SSMConfig(
            hidden_dim=self.hidden_dim,
            encoder_config=LinearEncoderConfig(in_features=self.in_features),
            sequence_mixer_configs=[
                LinOSSSequenceMixerConfig(state_dim=self.hidden_dim)
                for _ in range(self.num_blocks)
            ],
            block_configs=[
                LinOSSBlockConfig(drop_rate=self.drop_rate) for _ in range(self.num_blocks)
            ],
            head_config=ClassificationHeadConfig(out_features=self.out_features),
        )


class LinOSS(SSM):
    """LinOSS model."""

    def __init__(self, cfg: LinOSSConfig, key: PRNGKeyArray):
        super().__init__(cfg.build(), key)


if __name__ == "__main__":
    cfg = LinOSSConfig(
        in_features=784,
        hidden_dim=64,
        out_features=10,
        num_blocks=4,
        drop_rate=0.1,
    )

    model = LinOSS(cfg=cfg, key=jr.PRNGKey(0))
