"""Identity channel mixer."""

import equinox as eqx
from jaxtyping import Array


class IdentityChannelMixer(eqx.Module):
    """Identity channel mixer.

    This channel mixer simply returns the input unchanged.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """

    def __init__(self):
        """Initialize the identity channel mixer."""
        pass

    def __call__(self, x: Array) -> Array:
        """Forward pass of the identity channel mixer.

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged.
        """
        return x
