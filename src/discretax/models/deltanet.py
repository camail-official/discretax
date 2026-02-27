"""DeltaNet model."""

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.standard import StandardBlock
from discretax.channel_mixers.glu import GLU
from discretax.sequence_mixers.deltanet import DeltaNetSequenceMixer
from discretax.utils.config_mixin import PartialModule


class DeltaNet(eqx.nn.StatefulLayer, PartialModule):
    """DeltaNet model.

    This model implements stacked blocks with DeltaNet sequence mixers and GLU channel mixers.
    Use with eqx.nn.Sequential to compose with encoder and head.

    Attributes:
        blocks: List of standard blocks with DeltaNet sequence mixers.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from discretax.encoder import LinearEncoder
        from discretax.heads import ClassificationHead
        from discretax.models import DeltaNet

        key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
        model = DeltaNet(hidden_dim=64, num_blocks=4, key=keys[1])
        head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

        # Compose with Sequential
        full_model = eqx.nn.Sequential([encoder, model, head])
        ```

    Reference:
        DeltaNet: https://arxiv.org/abs/2406.06484
    """

    blocks: list[StandardBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        hidden_dim: int,
        num_blocks: int = 4,
        n_heads: int = 4,
        head_dim: int | None = None,
        chunk_size: int = 64,
        drop_rate: float = 0.1,
        prenorm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize the DeltaNet model.

        Args:
            key: JAX random key for initialization.
            hidden_dim: Hidden dimension for the model.
            num_blocks: Number of DeltaNet blocks to stack.
            n_heads: Number of attention heads in each DeltaNet sequence mixer.
            head_dim: Dimensionality per attention head. Defaults to
                `hidden_dim // n_heads`.
            chunk_size: Timesteps per chunk for the chunked delta rule. Must evenly
                divide the sequence length at inference time.
            drop_rate: Dropout rate for blocks.
            prenorm: Whether to apply prenorm in blocks.
            use_bias: Whether to use bias in GLU channel mixers and beta/out projections.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        keys = jr.split(key, 3 * num_blocks)

        self.blocks = []
        for i in range(num_blocks):
            seq_mixer = DeltaNetSequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                n_heads=n_heads,
                head_dim=head_dim,
                chunk_size=chunk_size,
                use_bias=use_bias,
            )

            chan_mixer = GLU(
                in_features=hidden_dim,
                key=keys[num_blocks + i],
                out_features=None,
                use_bias=use_bias,
            )

            block = StandardBlock(
                in_features=hidden_dim,
                sequence_mixer=seq_mixer,
                channel_mixer=chan_mixer,
                key=keys[2 * num_blocks + i],
                drop_rate=drop_rate,
                prenorm=prenorm,
            )
            self.blocks.append(block)

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass through the DeltaNet blocks.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        block_keys = jr.split(key, len(self.blocks))
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)
        return x, state
