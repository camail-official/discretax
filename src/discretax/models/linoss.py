"""LinOSS model."""

from typing import Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.standard import StandardBlock
from discretax.channel_mixers.glu import GLU
from discretax.sequence_mixers.linoss import LinOSSSequenceMixer
from discretax.utils.config_mixin import PartialModule


class LinOSS(eqx.nn.StatefulLayer, PartialModule):
    """LinOSS model.

    This model implements stacked blocks with LinOSS sequence mixers and GLU channel mixers.
    Use with eqx.nn.Sequential to compose with encoder and head.

    Attributes:
        blocks: List of standard blocks with LinOSS sequence mixers.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from discretax.encoder import LinearEncoder
        from discretax.heads import ClassificationHead
        from discretax.models import LinOSS

        key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
        model = LinOSS(hidden_dim=64, num_blocks=4, key=keys[1])
        head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

        # Compose with Sequential
        full_model = eqx.nn.Sequential([encoder, model, head])
        ```

    Reference:
        LinOSS: https://openreview.net/pdf?id=GRMfXcAAFh
    """

    blocks: list[StandardBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_dim: int,
        num_blocks: int = 4,
        state_dim: int = 64,
        discretization: Literal["IM", "IMEX"] = "IMEX",
        damping: bool = True,
        r_min: float = 0.9,
        theta_max: float = 3.14159265359,
        drop_rate: float = 0.1,
        prenorm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize the LinOSS model.

        Args:
            key: JAX random key for initialization.
            hidden_dim: hidden dimension for the model.
            num_blocks: number of LinOSS blocks to stack.
            state_dim: state space dimension for LinOSS sequence mixers.
            discretization: discretization method ("IM" or "IMEX").
            damping: whether to use damping in LinOSS.
            r_min: minimum value for the radius in LinOSS.
            theta_max: maximum value for theta parameter in LinOSS.
            drop_rate: dropout rate for blocks.
            prenorm: whether to apply prenorm in blocks.
            use_bias: whether to use bias in GLU channel mixers.
            **kwargs: additional keyword arguments.
        """
        keys = jr.split(key, 3 * num_blocks)

        # Build blocks with sequence mixers and channel mixers
        self.blocks = []
        for i in range(num_blocks):
            # Build sequence mixer
            seq_mixer = LinOSSSequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                state_dim=state_dim,
                discretization=discretization,
                damping=damping,
                r_min=r_min,
                theta_max=theta_max,
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
        """Forward pass through the LinOSS blocks.

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
    model = LinOSS(hidden_dim=64, num_blocks=4, key=keys[1])
    head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

    full_model = eqx.nn.Sequential([encoder, model, head])
    print(full_model)
