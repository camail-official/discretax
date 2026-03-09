"""GatedDeltaNet model.

This model stacks :class:`discretax.blocks.gated_deltanet.GatedDeltaNetBlock`
instances, each composed of a GatedDeltaNet sequence mixer and a SwiGLU
channel mixer, to follow the FLA-style architecture.
"""

from __future__ import annotations

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.gated_deltanet import GatedDeltaNetBlock
from discretax.channel_mixers.swi_glu import SwiGLU
from discretax.sequence_mixers.gated_deltanet import GatedDeltaNetSequenceMixer
from discretax.utils.config_mixin import PartialModule


class GatedDeltaNet(eqx.nn.StatefulLayer, PartialModule):
    """Stacked GatedDeltaNet blocks (FLA-style block wiring).

    Attributes:
        blocks: Ordered list of GatedDeltaNet blocks.
    """

    blocks: list[GatedDeltaNetBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        hidden_dim: int,
        num_blocks: int = 4,
        n_heads: int = 4,
        num_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        chunk_size: int = 64,
        attn_mode: str = "chunk",
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        hidden_ratio: float = 4.0,
        drop_rate: float = 0.0,
        norm_eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize the GatedDeltaNet model.

        Args:
            key: JAX PRNG key.
            hidden_dim: Hidden feature dimension.
            num_blocks: Number of stacked blocks.
            n_heads: Number of key/query heads.
            num_v_heads: Number of value heads. Defaults to `n_heads`.
            head_dim: Key/query head dimension. Defaults to `hidden_dim // n_heads`.
            expand_v: Value expansion factor per head.
            chunk_size: Chunk size configuration for the sequence mixer.
            attn_mode: Attention mode (`"chunk"` or `"fused_recurrent"`).
            use_gate: Whether to use gated output normalization in mixer.
            allow_neg_eigval: Whether to scale beta by 2.
            hidden_ratio: SwiGLU expansion ratio.
            drop_rate: Dropout rate inside each block.
            norm_eps: RMSNorm epsilon in block and mixer.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments forwarded to blocks.
        """
        del args
        keys = jr.split(key, 3 * num_blocks)

        self.blocks = []
        for i in range(num_blocks):
            seq_mixer = GatedDeltaNetSequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                n_heads=n_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
                expand_v=expand_v,
                chunk_size=chunk_size,
                mode=attn_mode,
                use_gate=use_gate,
                allow_neg_eigval=allow_neg_eigval,
                norm_eps=norm_eps,
            )

            chan_mixer = SwiGLU(
                in_features=hidden_dim,
                key=keys[num_blocks + i],
                hidden_ratio=hidden_ratio,
                use_bias=False,
            )

            block = GatedDeltaNetBlock(
                in_features=hidden_dim,
                sequence_mixer=seq_mixer,
                channel_mixer=chan_mixer,
                key=keys[2 * num_blocks + i],
                drop_rate=drop_rate,
                norm_eps=norm_eps,
                **kwargs,
            )
            self.blocks.append(block)

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Apply the stacked GatedDeltaNet blocks.

        Args:
            x: Input sequence tensor.
            state: Equinox state container.
            key: JAX PRNG key.

        Returns:
            Tuple of `(output, updated_state)`.
        """
        block_keys = jr.split(key, len(self.blocks))
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)
        return x, state
