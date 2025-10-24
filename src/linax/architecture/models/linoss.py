"""LinOSS model configuration."""

from dataclasses import dataclass, field

from linax.architecture.blocks.linoss import LinOSSBlockConfig
from linax.architecture.encoder import LinearEncoderConfig
from linax.architecture.heads.classification import ClassificationHeadConfig
from linax.architecture.models.ssm import SSMConfig
from linax.architecture.sequence_mixers.linoss import LinOSSSequenceMixerConfig


@dataclass
class LinOSSConfig(SSMConfig):
    """High-level configuration for LinOSS models.

    This is a simplified, user-friendly configuration that inherits from `SSMConfig`
    and only requires basic hyperparameters. It automatically composes the appropriate
    low-level components (encoder, sequence mixers, blocks, head) specific to the
    LinOSS architecture under the hood.

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
        model = config.build(key=key)
        ```

    Reference:
        LinOSS: https://arxiv.org/abs/2410.03943
    """

    # User-facing high-level parameters
    in_features: int
    hidden_dim: int
    out_features: int
    num_blocks: int = 4
    drop_rate: float = 0.1

    encoder_config: LinearEncoderConfig = field(init=False)
    sequence_mixer_configs: list[LinOSSSequenceMixerConfig] = field(init=False)
    block_configs: list[LinOSSBlockConfig] = field(init=False)
    head_config: ClassificationHeadConfig = field(init=False)

    def __post_init__(self):
        """Build component configs automatically from high-level parameters.

        Constructs LinOSS-specific components:
        - Linear encoder for input projection
        - LinOSS sequence mixers for temporal mixing
        - LinOSS blocks with GLU channel mixing
        - Classification head for output
        """
        # Build the low-level component configs automatically
        self.encoder_config = LinearEncoderConfig(in_features=self.in_features)
        self.sequence_mixer_configs = [
            LinOSSSequenceMixerConfig(state_dim=self.hidden_dim) for _ in range(self.num_blocks)
        ]
        self.block_configs = [
            LinOSSBlockConfig(drop_rate=self.drop_rate) for _ in range(self.num_blocks)
        ]
        self.head_config = ClassificationHeadConfig(out_features=self.out_features)

        # Call parent's __post_init__ for validation
        super().__post_init__()


if __name__ == "__main__":
    import jax.random as jr

    cfg = LinOSSConfig(
        in_features=784, hidden_dim=64, out_features=10, num_blocks=4, drop_rate=0.1
    )

    linoss = cfg.build(key=jr.PRNGKey(0))
    print(linoss)
