"""This module contains the heads implemented in Linax."""

from linax.architecture.heads.base import Head, HeadConfig
from linax.architecture.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from linax.architecture.heads.regression import (
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
