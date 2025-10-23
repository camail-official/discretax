"""Head base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class HeadConfig(ABC):
    """Configuration for heads."""

    out_features: int


class Head[ConfigType: HeadConfig](eqx.Module, ABC):
    """Abstract base class for all heads.

    Args:
        cfg:
          Configuration for the head.
        key:
          JAX random key for initialization.

    """

    @abstractmethod
    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
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
