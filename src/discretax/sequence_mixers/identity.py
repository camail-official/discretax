"""Sequence Mixer for Identity (pass-through).

This is a simple identity/pass-through sequence mixer that returns the input unchanged.
"""

from __future__ import annotations

from jaxtyping import Array, PRNGKeyArray

from discretax.sequence_mixers.base import SequenceMixer


class IdentitySequenceMixer(SequenceMixer):
    """Identity sequence mixer layer.

    This layer implements a simple identity/pass-through operation that returns
    the input sequence unchanged.

    Args:
        in_features: Input dimensionality.
        key: JAX random key for initialization.
    """

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ):
        """Initialize the Identity sequence mixer layer."""
        # Identity mixer has no parameters
        pass

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the Identity sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key (unused, for compatibility).

        Returns:
            The input sequence unchanged.
        """
        return x
