"""This module contains the heads implemented in Discretax."""

from discretax.heads.base import Head, HeadConfig
from discretax.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from discretax.heads.regression import (
    RegressionHead,
    RegressionHeadConfig,
)

__all__ = [
    "HeadConfig",
    "Head",
    "ClassificationHead",
    "ClassificationHeadConfig",
    "RegressionHead",
    "RegressionHeadConfig",
]
