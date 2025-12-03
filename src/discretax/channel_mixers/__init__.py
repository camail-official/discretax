"""This module contains the channel mixers implemented in Discretax."""

from discretax.channel_mixers.base import ChannelMixer, ChannelMixerConfig
from discretax.channel_mixers.glu import GLU, GLUConfig
from discretax.channel_mixers.identity import IdentityChannelMixer, IdentityChannelMixerConfig
from discretax.channel_mixers.mlp import MLPChannelMixer, MLPChannelMixerConfig
from discretax.channel_mixers.swi_glu import SwiGLU, SwiGLUConfig

__all__ = [
    "ChannelMixer",
    "ChannelMixerConfig",
    "GLU",
    "GLUConfig",
    "SwiGLU",
    "SwiGLUConfig",
    "IdentityChannelMixer",
    "IdentityChannelMixerConfig",
    "MLPChannelMixer",
    "MLPChannelMixerConfig",
]
