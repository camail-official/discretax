"""This module contains the heads implemented in Linax."""

from linax.heads.base import Head, HeadConfig
from linax.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from linax.heads.regression import (
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
