"""Base classes for linax."""

from dataclasses import dataclass


@dataclass
class AbstractConfig:
    """Abstract model configuration.

    This class defines the configuration for an abstract model in linax.

    Attributes:
        name:
          Name of the component (model, backbone, sequence mixer, etc.).
    """

    name: str = "component"
