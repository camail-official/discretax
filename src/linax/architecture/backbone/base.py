"""Backbone base class."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.base import AbstractConfig


@dataclass
class BackboneConfig(AbstractConfig):
    """Configuration for backbone models with no additional attributes."""

    pass


class Backbone[ConfigType: BackboneConfig](eqx.Module):
    """Abstract base class for all backbones.

    This class is used to define the interface for all backbones.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        in_features: int,
        key: PRNGKeyArray,
        **kwargs,
    ) -> None:
        """Initialize the backbone.

        Args:
            cfg:
              Configuration for the backbone.
            in_features:
              Dimensionality of the input features.
            key:
              JAX random key for initialization.
            **kwargs:
              Additional keyword arguments for specific backbone implementations.
        """
        pass

    @abstractmethod
    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the backbone.

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
        pass

    def filter_spec_lambda(self) -> Callable[..., bool]:
        """Filter specification for model parameters."""
        return lambda _: True

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Output features of the backbone.

        Returns:
            Number of output features.
        """
        pass
