"""LinOSS model."""

from dataclasses import dataclass, field

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.architecture.backbone.linoss import LinOSSBackbone, LinOSSBackboneConfig
from linax.architecture.models.base import AbstractModel, ModelConfig
from linax.architecture.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)


@dataclass
class LinOSSConfig(ModelConfig):
    """Configuration for the LinOSS model.

    This config uses a "single source of truth" pattern where the shared
    parameter (hidden_dim) is defined once and automatically propagated
    to component configs to ensure consistency.

    Attributes:
        name:
          Name of the model.
        hidden_dim:
          Hidden dimension shared between sequence mixer and backbone.
          This is the ONLY shared parameter and is automatically propagated.
        in_features:
          Dimensionality of the input features.
        out_features:
          Dimensionality of the output features.
        sequence_mixer_config:
          Configuration for the sequence mixer (hidden_dim will be set consistently).
        backbone_config:
          Configuration for the backbone (hidden_dim will be set consistently).
    """

    # All parameters must have defaults due to inheritance from AbstractConfig
    name: str = "linoss"
    hidden_dim: int = 64
    in_features: int = 64
    out_features: int | None = None

    # Component configs (hidden_dim will be propagated in __post_init__)
    sequence_mixer_config: LinOSSSequenceMixerConfig = field(
        default_factory=LinOSSSequenceMixerConfig
    )
    backbone_config: LinOSSBackboneConfig = field(default_factory=LinOSSBackboneConfig)

    def __post_init__(self):
        """Propagate shared parameter (hidden_dim) to component configs.

        This ensures consistency - hidden_dim is the only truly shared parameter
        that must match between sequence mixer and backbone. It's automatically
        set in both component configs to prevent dimension mismatches.

        Sets out_features to in_features if not specified.
        """
        # Set default out_features
        if self.out_features is None:
            self.out_features = self.in_features

        # Propagate shared hidden_dim to both configs
        self.sequence_mixer_config.hidden_dim = self.hidden_dim
        self.backbone_config.hidden_dim = self.hidden_dim


class LinOSS[ConfigType: LinOSSConfig](AbstractModel):
    """LinOSS model combining sequence mixer and backbone.

    Attributes:
        sequence_mixers:
          List of sequence mixer instances, one per block.
        backbone:
          The backbone instance containing blocks.
    """

    sequence_mixers: list[LinOSSSequenceMixer]
    backbone: LinOSSBackbone

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the LinOSS model.

        Args:
            cfg:
              Configuration for the model.
            key:
              JAX random key for initialization.
        """
        # Split keys for sequence mixers and backbone
        num_blocks = cfg.backbone_config.num_blocks
        keys = jr.split(key, num_blocks + 1)
        mixer_keys = keys[:-1]
        backbone_key = keys[-1]

        # Create independent sequence mixers for each block
        self.sequence_mixers = [
            LinOSSSequenceMixer(
                cfg=cfg.sequence_mixer_config,
                in_features=cfg.backbone_config.hidden_dim,
                key=mixer_key,
            )
            for mixer_key in mixer_keys
        ]

        # Create backbone with the sequence mixers
        self.backbone = LinOSSBackbone(
            cfg=cfg.backbone_config,
            in_features=cfg.in_features,
            key=backbone_key,
            sequence_mixers=self.sequence_mixers,
            out_features=cfg.out_features,
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
        return self.backbone(x, state, key)

    @property
    def out_features(self) -> int:
        """Output features of the model."""
        return self.backbone.out_features
