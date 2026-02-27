"""Identity channel mixer."""

from __future__ import annotations

from jaxtyping import Array, PRNGKeyArray

from discretax.channel_mixers.base import AbstractChannelMixer


class IdentityChannelMixer(AbstractChannelMixer):
    """Identity channel mixer.

    This channel mixer simply returns the input unchanged.
    """

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ):
        """Initialize the identity channel mixer.

        Args:
            in_features: the input dimensionality.
            key: JAX random key for initialization.
            *args: Additional arguments for the channel mixer.
            **kwargs: Additional keyword arguments for the channel mixer.
        """
        pass

    def __call__(self, x: Array) -> Array:
        """Forward pass of the identity channel mixer.

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged.
        """
        return x
