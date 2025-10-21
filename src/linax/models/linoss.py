"""LinOSS encoder."""

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
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
                dim=hidden_dim,
                discretization="IMEX",
                damping=True,
                r_min=0.9,
                theta_max=jnp.pi,
            ),
            in_features=hidden_dim,
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
        key, dropkey1, dropkey2 = jr.split(key, 3)
        skip = x
        x = self.sequence_mixer(x, key)
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

    hidden_dim: int = 64
    num_blocks: int = 4
    dropout_rate: float = 0.1
    name: Literal["linoss"] = "linoss"
    classification: bool = True
    # TODO: Add the sequence mixer config here (there should be a sensible default)


class LinossModel(AbstractModel[LinossModelConfig]):
    """LinOSS encoder."""

    linear_encoder: eqx.nn.Linear
    linear_decoder: eqx.nn.Linear
    blocks: list[LinossEncoderBlock]
    hidden_dim: int
    classification: bool

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        out_features: int | None = None,
        cfg: LinossModelConfig = LinossModelConfig(),
    ):
        """Initialize LinOSS model."""
        self.hidden_dim = cfg.hidden_dim
        self.classification = cfg.classification
        key, linear_encoder_key, linear_decoder_key, *block_keys = jr.split(
            key, cfg.num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(
            in_features,
            cfg.hidden_dim,
            key=linear_encoder_key,
            use_bias=False,
        )
        self.linear_decoder = eqx.nn.Linear(
            cfg.hidden_dim,
            in_features if out_features is None else out_features,
            key=linear_decoder_key,
            use_bias=False,
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
        y = jax.vmap(self.linear_encoder)(x)
        for i, (block, d_key) in enumerate(zip(self.blocks, dropout_keys)):
            y, state = block(y, state, key=d_key)

        x = jax.vmap(self.linear_decoder)(y)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.log_softmax(x, axis=-1)
        else:
            x = jnp.mean(x, axis=0)

        return x, state
