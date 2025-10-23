"""Encoder base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class EncoderConfig(ABC):
    """Configuration for encoders.

    Attributes:
        hidden_dim:
          Dimensionality of the hidden representation.
    """

    hidden_dim: int


class Encoder[ConfigType: EncoderConfig](eqx.Module, ABC):
    """Abstract base class for all encoders.

    This is the base class for all encoders.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
        """Initialize the encoder.

        Args:
            cfg:
              Configuration for the encoder.
            key:
              JAX random key for initialization.

        """
        pass

    @abstractmethod
    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the encoder.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        pass

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for encoder parameters."""
        return lambda _: True
