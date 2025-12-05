"""This module contains the sequence mixers implemented in Discretax."""

from discretax.sequence_mixers.base import SequenceMixer
from discretax.sequence_mixers.identity import IdentitySequenceMixer
from discretax.sequence_mixers.linoss import LinOSSSequenceMixer
from discretax.sequence_mixers.lru import LRUSequenceMixer
from discretax.sequence_mixers.s4d import S4DSequenceMixer
from discretax.sequence_mixers.s5 import S5SequenceMixer

__all__ = [
    "SequenceMixer",
    "IdentitySequenceMixer",
    "LinOSSSequenceMixer",
    "LRUSequenceMixer",
    "S4DSequenceMixer",
    "S5SequenceMixer",
]
