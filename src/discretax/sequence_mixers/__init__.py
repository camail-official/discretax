"""This module contains the sequence mixers implemented in Discretax."""

from discretax.sequence_mixers.base import (
    SequenceMixer,
    SequenceMixerConfig,
)
from discretax.sequence_mixers.identity import (
    IdentitySequenceMixer,
    IdentitySequenceMixerConfig,
)
from discretax.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)
from discretax.sequence_mixers.lru import (
    LRUSequenceMixer,
    LRUSequenceMixerConfig,
)
from discretax.sequence_mixers.s4d import (
    S4DSequenceMixer,
    S4DSequenceMixerConfig,
)
from discretax.sequence_mixers.s5 import (
    S5SequenceMixer,
    S5SequenceMixerConfig,
)

__all__ = [
    "SequenceMixer",
    "SequenceMixerConfig",
    "IdentitySequenceMixer",
    "IdentitySequenceMixerConfig",
    "LinOSSSequenceMixer",
    "LinOSSSequenceMixerConfig",
    "LRUSequenceMixer",
    "LRUSequenceMixerConfig",
    "S4DSequenceMixer",
    "S4DSequenceMixerConfig",
    "S5SequenceMixer",
    "S5SequenceMixerConfig",
]
