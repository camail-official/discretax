"""LinOSS encoder."""

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.layers.glu import GLU
from linax.models.base import AbstractModel, AbstractModelConfig
from linax.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)


class LinossEncoderBlock(eqx.Module):
    """LinOSS encoder block."""

    norm: eqx.nn.BatchNorm
    sequence_mixer: LinOSSSequenceMixer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        hidden_dim: int,
        drop_rate: float,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS encoder block."""
        sequence_mixer_key, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim,
            axis_name="batch",
            channelwise_affine=False,
            mode="ema",
        )
        self.sequence_mixer = LinOSSSequenceMixer(
            cfg=LinOSSSequenceMixerConfig(
                hidden_dim=hidden_dim,
                dropout_rate=drop_rate,
            ),
            key=sequence_mixer_key,
        )
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x = self.sequence_mixer(x)
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


@dataclass
class LinossModelConfig(AbstractModelConfig):
    """LinOSS model configuration."""

    name: Literal["linoss"]
    hidden_dim: int


class LinossModel(AbstractModel[LinossModelConfig]):
    """LinOSS encoder."""

    linear_encoder: eqx.nn.Linear
    blocks: list[LinossEncoderBlock]
    hidden_dim: int

    def __init__(
        self,
        cfg: LinossModelConfig,
        in_features: int,
        key: PRNGKeyArray,
    ):
        """Initialize LinOSS model."""
        key, linear_encoder_key, *block_keys = jr.split(key, cfg.num_blocks + 2)
        self.linear_encoder = eqx.nn.Linear(
            in_features,
            cfg.hidden_dim,
            key=linear_encoder_key,
        )

        key, *block_keys = jr.split(key, cfg.num_blocks + 1)
        self.blocks = [
            LinossEncoderBlock(
                cfg.hidden_dim,
                key=b_key,
                drop_rate=cfg.dropout_rate,
            )
            for b_key in block_keys
        ]

    @property
    def out_features(self) -> int:
        """Output features of the model."""
        return self.cfg.hidden_dim

    def __call__(self, x, state, key):
        """Forward pass of LinOSS model."""
        dropout_keys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for i, (block, d_key) in enumerate(zip(self.blocks, dropout_keys)):
            y, state = block(x, state, key=d_key)

        return y, state
