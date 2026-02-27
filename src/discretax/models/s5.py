"""S5 model."""

from typing import Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.standard import StandardBlock
from discretax.channel_mixers.glu import GLU
from discretax.sequence_mixers.s5 import S5SequenceMixer
from discretax.utils.config_mixin import PartialModule


class S5(eqx.nn.StatefulLayer, PartialModule):
    """S5 model.

    This model implements stacked blocks with S5 sequence mixers and GLU channel mixers.
    Use with eqx.nn.Sequential to compose with encoder and head.

    Attributes:
        blocks: List of standard blocks with S5 sequence mixers.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from discretax.encoder import LinearEncoder
        from discretax.heads import ClassificationHead
        from discretax.models import S5

        key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
        model = S5(hidden_dim=64, num_blocks=4, key=keys[1])
        head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

        # Compose with Sequential
        full_model = eqx.nn.Sequential([encoder, model, head])
        ```

    Reference:
        S5: https://openreview.net/pdf?id=Ai8Hw3AXqks
    """

    blocks: list[StandardBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        hidden_dim: int,
        num_blocks: int = 4,
        state_dim: int = 64,
        ssm_blocks: int = 1,
        C_init: Literal[
            "trunc_standard_normal", "lecun_normal", "complex_normal"
        ] = "lecun_normal",
        conj_sym: bool = True,
        clip_eigs: bool = True,
        discretization: Literal["zoh", "bilinear"] = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        step_rescale: float = 1.0,
        drop_rate: float = 0.1,
        prenorm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize the S5 model.

        Args:
            key: JAX random key for initialization.
            hidden_dim: hidden dimension for the model.
            num_blocks: number of S5 blocks to stack.
            state_dim: state space dimension for S5 sequence mixers.
            ssm_blocks: number of SSM blocks (for block-diagonal structure).
            C_init: initialization method for output matrix C.
            conj_sym: whether to enforce conjugate symmetry.
            clip_eigs: whether to clip eigenvalues to ensure stability.
            discretization: discretization method to use.
            dt_min: minimum discretization step size.
            dt_max: maximum discretization step size.
            step_rescale: rescaling factor for the discretization step.
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
            seq_mixer = S5SequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                state_dim=state_dim,
                ssm_blocks=ssm_blocks,
                C_init=C_init,
                conj_sym=conj_sym,
                clip_eigs=clip_eigs,
                discretization=discretization,
                dt_min=dt_min,
                dt_max=dt_max,
                step_rescale=step_rescale,
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
        """Forward pass through the S5 blocks.

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
    model = S5(hidden_dim=64, num_blocks=4, key=keys[1])
    head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

    full_model = eqx.nn.Sequential([encoder, model, head])
    print(full_model)
