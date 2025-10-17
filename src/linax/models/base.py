"""Model base class."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


@dataclass
class AbstractModelConfig:
    """Abstract model configuration."""

    name: str


ConfigType = TypeVar("ConfigType", bound=AbstractModelConfig)


class AbstractModel[ConfigType](eqx.Module):
    """Model base class."""

    cfg: ConfigType

    @abstractmethod
    def __init__(
        self,
        cfg: ConfigType,
        in_features: int,
        key: PRNGKeyArray,
    ):
        """Initialize the model."""

    @abstractmethod
    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the model."""

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Output features of the model."""
