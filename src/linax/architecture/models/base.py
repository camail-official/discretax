"""Model base class."""

from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.base import AbstractConfig


@dataclass
class ModelConfig(AbstractConfig):
    """Configuration for models with no additional attributes."""

    pass


class AbstractModel[ConfigType: AbstractConfig](eqx.Module):
    """Model base class.

    This class defines the base class for all models in linax.
    """

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        in_features: int,
        key: PRNGKeyArray,
    ):
        """Initialize the model.

        Args:
            cfg:
              Configuration for the model.
            in_features:
              Dimensionality of the input features.
            key:
              JAX random key for initialization.
        """

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

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Output features of the model.

        This property returns the number of output features of the model.

        Returns:
            Number of output features.
        """
