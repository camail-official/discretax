"""Tests for channel mixers and base build semantics."""

import jax.numpy as jnp
import jax.random as jr

from discretax.channel_mixers.glu import GLU
from discretax.channel_mixers.identity import IdentityChannelMixer
from discretax.channel_mixers.mlp import MLPChannelMixer


def test_glu_build_and_call_shapes():
    """Test GLU channel mixer builds correctly and produces expected output shapes.

    This test verifies that:
    1. GLU instantiates correctly
    2. The GLU mixer accepts input of shape (in_features,) and outputs (out_features,)
    3. Direct instantiation works as expected
    """
    key = jr.PRNGKey(0)
    in_features = 8
    out_features = 8

    mixer = GLU(in_features=in_features, out_features=out_features, key=key)
    assert isinstance(mixer, GLU)

    x = jr.normal(key, (in_features,))
    y = mixer(x)
    assert y.shape == (out_features,)


def test_mlp_build_and_call_shapes():
    """Test MLP channel mixer builds correctly and produces expected output shapes.

    This test verifies that:
    1. MLPChannelMixer instantiates correctly
    2. The MLP mixer can handle different input and output dimensions
    3. Direct instantiation with out_features parameter works correctly
    """
    key = jr.PRNGKey(1)
    in_features = 8
    out_features = 16

    mixer = MLPChannelMixer(in_features=in_features, out_features=out_features, key=key)
    assert isinstance(mixer, MLPChannelMixer)

    x = jr.normal(key, (in_features,))
    y = mixer(x)
    assert y.shape == (out_features,)


def test_identity_build_and_call_passthrough():
    """Test identity channel mixer builds correctly and acts as a passthrough.

    This test verifies that:
    1. IdentityChannelMixer instantiates correctly
    2. The identity mixer returns input unchanged (passthrough behavior)
    3. Direct instantiation works with out_features=None (defaults to in_features)
    4. Output shape matches input shape exactly
    """
    key = jr.PRNGKey(2)
    in_features = 8

    mixer = IdentityChannelMixer(in_features=in_features, out_features=None, key=key)
    assert isinstance(mixer, IdentityChannelMixer)

    x = jnp.arange(in_features).astype(jnp.float32)
    y = mixer(x)
    assert y.shape == (in_features,)
    assert jnp.allclose(x, y)
