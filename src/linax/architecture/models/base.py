"""Model base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass
class ModelConfig(ABC):
    """Configuration for models."""


class AbstractModel[ConfigType: ModelConfig](eqx.Module, ABC):
    """Model base class.

    This class defines the base class for all models in linax.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            cfg:
              Configuration for the model.
            key:
              JAX random key for initialization.
            **kwargs:
              Additional keyword arguments.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the model.

        This method implements the forward pass of the model.

        Args:
            x:
              Input tensor.
            state:
              Current state for stateful layers.
            key:
              JAX random key for initialization.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        pass
