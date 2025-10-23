"""Head base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass
class HeadConfig(ABC):
    """Configuration for heads."""

    name: str
    in_features: int
    out_features: int


class Head[ConfigType: HeadConfig](eqx.Module, ABC):
    """Abstract base class for all heads."""

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
        """Initialize the head.

        Args:
            cfg:
              Configuration for the head.
            key:
              JAX random key for initialization.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the head.

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
        """Filter specification for head parameters."""
        return lambda _: True
