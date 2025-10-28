"""Identity channel mixer."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Array, PRNGKeyArray

from linax.channel_mixers.base import ChannelMixer, ChannelMixerConfig


@dataclass(frozen=True)
class IdentityChannelMixerConfig(ChannelMixerConfig):
    """Configuration for the identity channel mixer."""

    def build(
        self, in_features: int, out_features: int | None, key: PRNGKeyArray
    ) -> IdentityChannelMixer:
        """Build IdentityChannelMixer from config.

        Args:
            in_features: Input dimensionality.
            out_features: Optional output dimensionality. If None, defaults to in_features.
            key: JAX random key for initialization.

        Returns:
            The IdentityChannelMixer instance.
        """
        return IdentityChannelMixer(
            in_features=in_features, cfg=self, key=key, out_features=out_features
        )


class IdentityChannelMixer[ConfigType: IdentityChannelMixerConfig](ChannelMixer):
    """Identity channel mixer.

    This channel mixer simply returns the input unchanged.

    Args:
        in_features: The input dimensionality.
        cfg: Configuration for the identity channel mixer.
        key: JAX random key for initialization.
        out_features: Optional output dimensionality. If None, defaults to in_features.
    """

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
        *,
        out_features: int | None = None,
    ):
        """Initialize the identity channel mixer."""

    def __call__(self, x: Array) -> Array:
        """Forward pass of the identity channel mixer.

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged.
        """
        return x
