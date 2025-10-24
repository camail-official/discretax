"""Sequence Mixer for Identity (pass-through).

This is a simple identity/pass-through sequence mixer that returns the input unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixer, SequenceMixerConfig


@dataclass(frozen=True)
class IdentitySequenceMixerConfig(SequenceMixerConfig):
    """Configuration for the Identity sequence mixer.

    This configuration class defines a simple pass-through sequence mixer that
    returns the input unchanged. The state_dim parameter is required for compatibility
    with the base class but is not used by the identity mixer.

    Attributes:
        state_dim: Dimensionality of the state space (unused, for compatibility).
    """

    state_dim: int = 0

    def build(self, in_features: int, key: PRNGKeyArray) -> IdentitySequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """
        return IdentitySequenceMixer(in_features=in_features, cfg=self, key=key)


class IdentitySequenceMixer[ConfigType: IdentitySequenceMixerConfig](SequenceMixer):
    """Identity sequence mixer layer.

    This layer implements a simple identity/pass-through operation that returns
    the input sequence unchanged.

    Attributes:
        None (stateless pass-through operation).

    Args:
        in_features: Input dimensionality.
        cfg: Configuration for the Identity sequence mixer.
        key: JAX random key for initialization.
        **kwargs: Additional keyword arguments (unused, for compatibility).
    """

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
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
