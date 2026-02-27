"""LRU model."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.standard import StandardBlock
from discretax.channel_mixers.glu import GLU
from discretax.sequence_mixers.lru import LRUSequenceMixer
from discretax.utils.config_mixin import PartialModule


class LRU(eqx.nn.StatefulLayer, PartialModule):
    """LRU model.

    This model implements stacked blocks with LRU sequence mixers and GLU channel mixers.
    Use with eqx.nn.Sequential to compose with encoder and head.

    Attributes:
        blocks: List of standard blocks with LRU sequence mixers.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from discretax.encoder import LinearEncoder
        from discretax.heads import ClassificationHead
        from discretax.models import LRU

        key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
        model = LRU(hidden_dim=64, num_blocks=4, key=keys[1])
        head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

        # Compose with Sequential
        full_model = eqx.nn.Sequential([encoder, model, head])
        ```

    Reference:
        LRU: https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf
    """

    blocks: list[StandardBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        hidden_dim: int,
        num_blocks: int = 4,
        state_dim: int = 64,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * jnp.pi,
        drop_rate: float = 0.1,
        prenorm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize the LRU model.

        Args:
            key: JAX random key for initialization.
            hidden_dim: hidden dimension for the model.
            num_blocks: number of LRU blocks to stack.
            state_dim: state space dimension for LRU sequence mixers.
            r_min: minimum radius for complex-valued eigenvalues.
            r_max: maximum radius for complex-valued eigenvalues.
            max_phase: maximum phase angle for complex-valued eigenvalues.
            drop_rate: dropout rate for blocks.
            prenorm: whether to apply prenorm in blocks.
            use_bias: whether to use bias in GLU channel mixers.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        keys = jr.split(key, 3 * num_blocks)

        # Build blocks with sequence mixers and channel mixers
        self.blocks = []
        for i in range(num_blocks):
            # Build sequence mixer
            seq_mixer = LRUSequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                state_dim=state_dim,
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
            )

            # Build channel mixer
            chan_mixer = GLU(
                in_features=hidden_dim,
                key=keys[num_blocks + i],
                out_features=None,
                use_bias=use_bias,
            )

            # Build block
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
        """Forward pass through the LRU blocks.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        # Prepare the keys
        block_keys = jr.split(key, len(self.blocks))

        # Apply the blocks
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)

        return x, state


if __name__ == "__main__":
    import jax.random as jr

    from discretax.encoder import LinearEncoder
    from discretax.heads import ClassificationHead

    key = jr.PRNGKey(0)
    keys = jr.split(key, 3)

    encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
    model = LRU(hidden_dim=64, num_blocks=4, key=keys[1])
    head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

    full_model = eqx.nn.Sequential([encoder, model, head])
    print(full_model)
