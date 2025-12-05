"""This module contains the channel mixers implemented in Discretax."""

from discretax.channel_mixers.base import ChannelMixer
from discretax.channel_mixers.glu import GLU
from discretax.channel_mixers.identity import IdentityChannelMixer
from discretax.channel_mixers.mlp import MLPChannelMixer
from discretax.channel_mixers.swi_glu import SwiGLU

__all__ = [
    "ChannelMixer",
    "GLU",
    "SwiGLU",
    "IdentityChannelMixer",
    "MLPChannelMixer",
]
