"""Unit tests for the DeltaNet ops, sequence mixer, and model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from discretax.encoder import LinearEncoder
from discretax.heads.classification import ClassificationHead
from discretax.models.deltanet import DeltaNet
from discretax.ops.delta_rule import chunk_delta_rule, chunk_delta_rule_head
from discretax.sequence_mixers.deltanet import DeltaNetSequenceMixer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qkvb(key, timesteps, n_heads, head_dim):
    """Generate normalised Q, K, V and sigmoid beta for op tests."""
    k1, k2, k3, k4 = jr.split(key, 4)
    Q = jr.normal(k1, (timesteps, n_heads, head_dim))
    K = jr.normal(k2, (timesteps, n_heads, head_dim))
    V = jr.normal(k3, (timesteps, n_heads, head_dim))
    beta = jax.nn.sigmoid(jr.normal(k4, (timesteps,)))
    return Q, K, V, beta


# ---------------------------------------------------------------------------
# chunk_delta_rule_head
# ---------------------------------------------------------------------------


def test_chunk_delta_rule_head_output_shape():
    """chunk_delta_rule_head returns shape (timesteps, head_dim)."""
    timesteps, head_dim, chunk_size = 8, 8, 4
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(0), 4)
    Q = jr.normal(k1, (timesteps, head_dim))
    K = jr.normal(k2, (timesteps, head_dim))
    V = jr.normal(k3, (timesteps, head_dim))
    beta = jax.nn.sigmoid(jr.normal(k4, (timesteps,)))

    O = chunk_delta_rule_head(Q, K, V, beta, chunk_size)
    assert O.shape == (timesteps, head_dim)


def test_chunk_delta_rule_head_output_is_finite():
    """chunk_delta_rule_head output contains no NaNs or Infs."""
    timesteps, head_dim, chunk_size = 8, 8, 4
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(1), 4)
    Q = jr.normal(k1, (timesteps, head_dim))
    K = jr.normal(k2, (timesteps, head_dim))
    V = jr.normal(k3, (timesteps, head_dim))
    beta = jax.nn.sigmoid(jr.normal(k4, (timesteps,)))

    O = chunk_delta_rule_head(Q, K, V, beta, chunk_size)
    assert jnp.all(jnp.isfinite(O))


# ---------------------------------------------------------------------------
# chunk_delta_rule
# ---------------------------------------------------------------------------


def test_chunk_delta_rule_output_shape():
    """chunk_delta_rule returns shape (timesteps, n_heads, head_dim)."""
    timesteps, n_heads, head_dim, chunk_size = 8, 2, 4, 4
    Q, K, V, beta = _qkvb(jr.PRNGKey(2), timesteps, n_heads, head_dim)

    O = chunk_delta_rule(Q, K, V, beta, chunk_size)
    assert O.shape == (timesteps, n_heads, head_dim)


def test_chunk_delta_rule_output_is_finite():
    """chunk_delta_rule output contains no NaNs or Infs."""
    timesteps, n_heads, head_dim, chunk_size = 8, 2, 4, 4
    Q, K, V, beta = _qkvb(jr.PRNGKey(3), timesteps, n_heads, head_dim)

    O = chunk_delta_rule(Q, K, V, beta, chunk_size)
    assert jnp.all(jnp.isfinite(O))


def test_chunk_delta_rule_single_head_matches_head_fn():
    """chunk_delta_rule with n_heads=1 matches chunk_delta_rule_head directly."""
    timesteps, head_dim, chunk_size = 8, 4, 4
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(4), 4)
    Q = jr.normal(k1, (timesteps, head_dim))
    K = jr.normal(k2, (timesteps, head_dim))
    V = jr.normal(k3, (timesteps, head_dim))
    beta = jax.nn.sigmoid(jr.normal(k4, (timesteps,)))

    O_head = chunk_delta_rule_head(Q, K, V, beta, chunk_size)
    O_multi = chunk_delta_rule(Q[:, None, :], K[:, None, :], V[:, None, :], beta, chunk_size)
    assert jnp.allclose(O_head, O_multi[:, 0, :], atol=1e-5)


# ---------------------------------------------------------------------------
# DeltaNetSequenceMixer
# ---------------------------------------------------------------------------


def test_deltanet_mixer_builds():
    """DeltaNetSequenceMixer instantiates without errors."""
    mixer = DeltaNetSequenceMixer(
        in_features=16, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    assert isinstance(mixer, DeltaNetSequenceMixer)


def test_deltanet_mixer_output_shape():
    """DeltaNetSequenceMixer returns (timesteps, in_features)."""
    in_features, timesteps = 16, 8
    mixer = DeltaNetSequenceMixer(
        in_features=in_features, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    x = jr.normal(jr.PRNGKey(1), (timesteps, in_features))
    y = mixer(x, jr.PRNGKey(2))
    assert y.shape == (timesteps, in_features)


def test_deltanet_mixer_default_head_dim():
    """head_dim defaults to in_features // n_heads when omitted."""
    in_features, n_heads = 16, 4
    mixer = DeltaNetSequenceMixer(
        in_features=in_features, key=jr.PRNGKey(0), n_heads=n_heads, chunk_size=4
    )
    assert mixer.head_dim == in_features // n_heads


def test_deltanet_mixer_chunk_size_assertion():
    """AssertionError is raised when timesteps is not divisible by chunk_size."""
    mixer = DeltaNetSequenceMixer(
        in_features=16, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    x = jr.normal(jr.PRNGKey(1), (7, 16))  # 7 % 4 != 0
    with pytest.raises(ValueError, match="chunk_size"):
        mixer(x, jr.PRNGKey(2))


def test_deltanet_mixer_output_is_finite():
    """DeltaNetSequenceMixer output contains no NaNs or Infs."""
    in_features, timesteps = 16, 8
    mixer = DeltaNetSequenceMixer(
        in_features=in_features, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    x = jr.normal(jr.PRNGKey(1), (timesteps, in_features))
    y = mixer(x, jr.PRNGKey(2))
    assert jnp.all(jnp.isfinite(y))


def test_deltanet_mixer_deterministic():
    """DeltaNetSequenceMixer produces identical outputs for the same inputs."""
    in_features, timesteps = 16, 8
    mixer = DeltaNetSequenceMixer(
        in_features=in_features, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    x = jr.normal(jr.PRNGKey(1), (timesteps, in_features))
    y1 = mixer(x, jr.PRNGKey(2))
    y2 = mixer(x, jr.PRNGKey(2))
    assert jnp.allclose(y1, y2)


def test_deltanet_mixer_different_inputs_differ():
    """DeltaNetSequenceMixer produces different outputs for different inputs."""
    in_features, timesteps = 16, 8
    mixer = DeltaNetSequenceMixer(
        in_features=in_features, key=jr.PRNGKey(0), n_heads=2, head_dim=8, chunk_size=4
    )
    x1 = jr.normal(jr.PRNGKey(1), (timesteps, in_features))
    x2 = jr.normal(jr.PRNGKey(2), (timesteps, in_features))
    y1 = mixer(x1, jr.PRNGKey(3))
    y2 = mixer(x2, jr.PRNGKey(3))
    assert not jnp.allclose(y1, y2)


# ---------------------------------------------------------------------------
# DeltaNet model
# ---------------------------------------------------------------------------


def test_deltanet_model_forward():
    """DeltaNet end-to-end forward pass with encoder and classification head.

    This test verifies that:
    1. DeltaNet builds successfully with the given hyperparameters.
    2. Forward pass composed with encoder and head produces the correct output shape.
    3. Batching via vmap works correctly.
    """
    key = jr.PRNGKey(0)
    k_enc, k_model, k_head = jr.split(key, 3)

    in_features, hidden_dim, n_classes = 16, 16, 3
    batch_size, timesteps, chunk_size = 2, 16, 4

    encoder = LinearEncoder(in_features=in_features, out_features=hidden_dim, key=k_enc)
    model = DeltaNet(
        hidden_dim=hidden_dim,
        num_blocks=2,
        n_heads=2,
        head_dim=8,
        chunk_size=chunk_size,
        drop_rate=0.0,
        key=k_model,
    )
    head = ClassificationHead(in_features=hidden_dim, out_features=n_classes, key=k_head)

    full_model = eqx.nn.Sequential([encoder, model, head])

    x = jr.normal(jr.PRNGKey(1), (batch_size, timesteps, in_features))
    state = eqx.nn.State(full_model)

    def single_forward(x_single, key_single):
        return full_model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(2), batch_size))

    assert y.shape == (batch_size, n_classes)


def test_deltanet_model_output_is_finite():
    """DeltaNet model output contains no NaNs or Infs."""
    key = jr.PRNGKey(5)
    k_enc, k_model, k_head = jr.split(key, 3)

    in_features, hidden_dim, n_classes = 16, 16, 3
    batch_size, timesteps, chunk_size = 2, 16, 4

    encoder = LinearEncoder(in_features=in_features, out_features=hidden_dim, key=k_enc)
    model = DeltaNet(
        hidden_dim=hidden_dim,
        num_blocks=2,
        n_heads=2,
        head_dim=8,
        chunk_size=chunk_size,
        drop_rate=0.0,
        key=k_model,
    )
    head = ClassificationHead(in_features=hidden_dim, out_features=n_classes, key=k_head)

    full_model = eqx.nn.Sequential([encoder, model, head])
    x = jr.normal(jr.PRNGKey(6), (batch_size, timesteps, in_features))
    state = eqx.nn.State(full_model)

    def single_forward(x_single, key_single):
        return full_model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(7), batch_size))

    assert jnp.all(jnp.isfinite(y))
