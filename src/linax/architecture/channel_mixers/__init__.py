"""This module contains the channel mixers implemented in Linax."""

from linax.architecture.channel_mixers.glu import GLU
from linax.architecture.channel_mixers.swi_glu import SwiGLU

__all__ = ["GLU", "SwiGLU"]
