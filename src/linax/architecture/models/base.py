"""This module contains the base class for all models in Linax."""

from abc import ABC
from dataclasses import dataclass


@dataclass
class ModelConfig(ABC):
    """Configuration class for models."""
