"""LinOSS model."""

from dataclasses import dataclass

import jax.random as jr
from jaxtyping import PRNGKeyArray

from linax.architecture.blocks.linoss import LinOSSBlockConfig
from linax.architecture.encoder import LinearEncoderConfig
from linax.architecture.heads.classification import ClassificationHeadConfig
from linax.architecture.models.base import ModelConfig
from linax.architecture.models.ssm import SSM, SSMConfig
from linax.architecture.sequence_mixers.linoss import LinOSSSequenceMixerConfig


@dataclass
class LinOSSConfig(ModelConfig):
    """High-level configuration for LinOSS models.

    This is a simplified, user-friendly configuration that only requires basic
    hyperparameters. It automatically composes the appropriate low-level components
    (encoder, sequence mixers, blocks, head) specific to the LinOSS architecture.

    Use this when:
    - You want to quickly build a LinOSS model
    - You only need to tune key hyperparameters
    - You want sensible defaults for LinOSS-specific components

    Attributes:
        in_features:
          Dimensionality of the input features.
        hidden_dim:
          Dimensionality of the hidden state throughout the model.
        out_features:
          Dimensionality of the output (e.g., number of classes).
        num_blocks:
          Number of LinOSS blocks to stack.
        drop_rate:
          Dropout rate applied in the blocks.

    Example:
        ```python
        # Simple high-level usage
        config = LinOSSConfig(
            in_features=784,
            hidden_dim=64,
            out_features=10,
            num_blocks=4,
            drop_rate=0.1,
        )
        model = LinOSS(cfg=config, key=key)
        ```

    Reference:
        LinOSS: https://arxiv.org/abs/2410.03943
    """

    in_features: int = 784
    hidden_dim: int = 20
    out_features: int = 10
    num_blocks: int = 4
    drop_rate: float = 0.1

    def build_ssm_config(self) -> SSMConfig:
        """Build the corresponding low-level SSMConfig.

        Constructs an `SSMConfig` with LinOSS-specific components:
        - Linear encoder for input projection
        - LinOSS sequence mixers for temporal mixing
        - LinOSS blocks with GLU channel mixing
        - Classification head for output

        Returns:
            Low-level SSMConfig ready to instantiate an SSM model.
        """
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
    """LinOSS model.

    This is a convenience class that constructs an SSM with LinOSS-specific
    components. It inherits all functionality from `SSM` but provides a
    simpler configuration interface via `LinOSSConfig`.

    Args:
        cfg:
          High-level LinOSSConfig specifying key hyperparameters.
        key:
          JAX random key for parameter initialization.
    """

    def __init__(self, cfg: LinOSSConfig, key: PRNGKeyArray):
        super().__init__(cfg.build_ssm_config(), key)


if __name__ == "__main__":
    cfg = LinOSSConfig(
        in_features=784,
        hidden_dim=64,
        out_features=10,
        num_blocks=4,
        drop_rate=0.1,
    )

    model = LinOSS(cfg=cfg, key=jr.PRNGKey(0))
