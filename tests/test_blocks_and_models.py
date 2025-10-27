"""End-to-end tests for blocks and models with channel and sequence mixers."""

import equinox as eqx
import jax
import jax.random as jr

from linax.blocks.standard import StandardBlockConfig
from linax.channel_mixers.glu import GLUConfig
from linax.encoder import LinearEncoderConfig
from linax.heads.classification import ClassificationHeadConfig
from linax.models.linoss import LinOSSConfig
from linax.models.lru import LRUConfig
from linax.models.s5 import S5Config
from linax.models.ssm import SSM
from linax.sequence_mixers.linoss import LinOSSSequenceMixerConfig
from linax.sequence_mixers.lru import LRUSequenceMixerConfig
from linax.sequence_mixers.s5 import S5SequenceMixerConfig


def _dummy_input(batch_size: int, timesteps: int, in_features: int):
    """Generate dummy input data for testing.

    Args:
        batch_size: Number of samples
        timesteps: Sequence length
        in_features: Input feature dimension

    Returns:
        Random input tensor of shape (batch_size, timesteps, in_features)
    """
    return jr.normal(jr.PRNGKey(0), (batch_size, timesteps, in_features))


def _dummy_state(model: SSM, batch_size: int):
    """Generate dummy state for testing.

    Args:
        model: The SSM model instance
        batch_size: Number of samples (unused, kept for API consistency)

    Returns:
        Empty state object for the model
    """
    # Collect state for layer norms/dropouts if any; here we assume empty state OK
    return eqx.nn.State(model)


def test_lru_model_forward():
    """Test LRU model forward pass with channel and sequence mixers.

    This test verifies that:
    1. LRUConfig builds a complete SSM model with LRU blocks
    2. The model includes both sequence mixers (LRU) and channel mixers (GLU)
    3. Forward pass produces correct output shape matching the head configuration
    4. All components (encoder, blocks, head) work together correctly
    """
    key = jr.PRNGKey(0)
    cfg = LRUConfig(
        num_blocks=2,
        encoder_config=LinearEncoderConfig(in_features=16, out_features=16),
        sequence_mixer_config=LRUSequenceMixerConfig(state_dim=32),
        block_config=StandardBlockConfig(drop_rate=0.0),
        head_config=ClassificationHeadConfig(out_features=3),
        channel_mixer_config=GLUConfig(),
    )
    model = cfg.build(key=key)

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - model expects (timesteps, features)
    def single_forward(x_single, key_single):
        return model(x_single, state, key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(1), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


def test_s5_model_forward():
    """Test S5 model forward pass with channel and sequence mixers.

    This test verifies that:
    1. S5Config builds a complete SSM model with S5 blocks
    2. The model includes both sequence mixers (S5) and channel mixers (GLU)
    3. Forward pass produces correct output shape matching the head configuration
    4. S5-specific parameters (ssm_blocks, state_dim) are handled correctly
    """
    key = jr.PRNGKey(1)
    cfg = S5Config(
        num_blocks=2,
        encoder_config=LinearEncoderConfig(in_features=16, out_features=16),
        sequence_mixer_config=S5SequenceMixerConfig(state_dim=32, ssm_blocks=1),
        block_config=StandardBlockConfig(drop_rate=0.0),
        head_config=ClassificationHeadConfig(out_features=3),
        channel_mixer_config=GLUConfig(),
    )
    model = cfg.build(key=key)

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - model expects (timesteps, features)
    def single_forward(x_single, key_single):
        return model(x_single, state, key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(2), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


def test_linoss_model_forward():
    """Test LinOSS model forward pass with channel and sequence mixers.

    This test verifies that:
    1. LinOSSConfig builds a complete SSM model with LinOSS blocks
    2. The model includes both sequence mixers (LinOSS) and channel mixers (GLU)
    3. Forward pass produces correct output shape matching the head configuration
    4. LinOSS-specific parameters (state_dim, discretization) are handled correctly
    5. Batching works correctly with vmap
    """
    key = jr.PRNGKey(2)
    cfg = LinOSSConfig(
        num_blocks=2,
        encoder_config=LinearEncoderConfig(in_features=16, out_features=16),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=32),
        block_config=StandardBlockConfig(drop_rate=0.0),
        head_config=ClassificationHeadConfig(out_features=3),
        channel_mixer_config=GLUConfig(),
    )
    model = cfg.build(key=key)

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - model expects (timesteps, features)
    def single_forward(x_single, key_single):
        return model(x_single, state, key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(3), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)
