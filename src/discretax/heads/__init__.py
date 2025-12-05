"""This module contains the heads implemented in Discretax."""

from discretax.heads.base import Head
from discretax.heads.classification import ClassificationHead
from discretax.heads.regression import RegressionHead

__all__ = [
    "Head",
    "ClassificationHead",
    "RegressionHead",
]
