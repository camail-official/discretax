"""Test channel mixers (MLP and Identity) to verify they work correctly."""

import jax
import jax.numpy as jnp
import jax.random as jr
from src.linax.channel_mixers.identity import IdentityChannelMixer
from src.linax.channel_mixers.mlp import MLPChannelMixer, non_linearity_factory


def test_non_linearity_factory():
    """Test the non-linearity factory function."""
    print("\n" + "=" * 70)
    print("Testing Non-linearity Factory")
    print("=" * 70)

    # Test all supported activations
    activations = ["relu", "gelu", "swish", "silu", "tanh"]

    for activation_name in activations:
        print(f"\nTesting {activation_name}...")

        # Get the activation function
        activation_func = non_linearity_factory(activation_name)
        print(f"✓ Created {activation_name} function")

        # Test with dummy data
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = activation_func(x)

        print(f"  Input:  {x}")
        print(f"  Output: {y}")

        # Verify output is valid
        assert not jnp.any(jnp.isnan(y)), f"{activation_name} produced NaN!"
        print(f"✓ {activation_name} works correctly")

    # Test invalid activation
    print("\nTesting invalid activation...")
    try:
        non_linearity_factory("invalid")
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ Non-linearity factory test passed!")
    return True


def test_mlp_channel_mixer():
    """Test MLP channel mixer."""
    print("\n" + "=" * 70)
    print("Testing MLP Channel Mixer")
    print("=" * 70)

    # Configuration
    input_dim = 16
    output_dim = 32
    seq_len = 10

    key = jr.PRNGKey(0)

    results = {}

    # Test each activation function
    activations = ["relu", "gelu", "swish", "silu", "tanh"]

    for activation in activations:
        try:
            print(f"\n{'=' * 70}")
            print(f"Testing MLP with {activation}")
            print(f"{'=' * 70}")

            # Create MLP
            mlp = MLPChannelMixer(
                input_dim=input_dim,
                output_dim=output_dim,
                non_linearity=activation,
                use_bias=True,
                key=key,
            )
            print(f"✓ MLP created with {activation}")

            # Create dummy input
            x = jr.normal(key, (seq_len, input_dim))
            print(f"\nInput shape: {x.shape}")
            print(f"Input stats - mean: {jnp.mean(x):.4f}, std: {jnp.std(x):.4f}")

            # Forward pass (vmap over sequence dimension, as done in blocks)
            y = jax.vmap(mlp)(x)
            print(f"\nOutput shape: {y.shape}")
            print(f"Output stats - mean: {jnp.mean(y):.4f}, std: {jnp.std(y):.4f}")

            # Verify output shape
            expected_shape = (seq_len, output_dim)
            assert y.shape == expected_shape, f"Shape mismatch: {y.shape} != {expected_shape}"
            print(f"✓ Output shape correct: {y.shape}")

            # Verify output is valid
            assert not jnp.any(jnp.isnan(y)), "Output contains NaN!"
            print("✓ Output contains no NaN")

            # For ReLU, verify non-negativity
            if activation == "relu":
                assert jnp.all(y >= 0), "ReLU output contains negative values!"
                print("✓ ReLU output is non-negative")

            # For tanh, verify output is in [-1, 1]
            if activation == "tanh":
                assert jnp.all(jnp.abs(y) <= 1.0), "Tanh output outside [-1, 1]!"
                print("✓ Tanh output in [-1, 1]")

            results[f"MLP-{activation}"] = "✓ PASSED"

        except Exception as e:
            print(f"\n✗ MLP with {activation} FAILED: {e}")
            import traceback

            traceback.print_exc()
            results[f"MLP-{activation}"] = f"✗ FAILED: {e}"

    # Print summary
    print("\n" + "=" * 70)
    print("MLP TEST SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"{name:20s}: {status}")

    all_passed = all("PASSED" in status for status in results.values())
    return all_passed


def test_identity_channel_mixer():
    """Test Identity channel mixer."""
    print("\n" + "=" * 70)
    print("Testing Identity Channel Mixer")
    print("=" * 70)

    # Configuration
    input_dim = 16
    seq_len = 10

    key = jr.PRNGKey(0)

    # Create Identity mixer
    identity = IdentityChannelMixer()
    print("✓ Identity channel mixer created")

    # Create dummy input
    x = jr.normal(key, (seq_len, input_dim))
    print(f"\nInput shape: {x.shape}")
    print(f"Input stats - mean: {jnp.mean(x):.4f}, std: {jnp.std(x):.4f}")

    # Forward pass
    y = identity(x)
    print(f"\nOutput shape: {y.shape}")
    print(f"Output stats - mean: {jnp.mean(y):.4f}, std: {jnp.std(y):.4f}")

    # Verify output shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f"✓ Output shape correct: {y.shape}")

    # Verify output equals input (identity property)
    assert jnp.allclose(y, x), "Identity output does not equal input!"
    print("✓ Identity property verified (output == input)")

    # Verify exact equality
    assert jnp.array_equal(y, x), "Identity output is not exactly equal to input!"
    print("✓ Exact equality verified")

    print("\n✓ Identity channel mixer test passed!")
    return True


def main():
    """Run all channel mixer tests."""
    print("\n" + "=" * 70)
    print("TESTING ALL CHANNEL MIXERS")
    print("=" * 70)

    results = {}

    # Test 1: Non-linearity Factory
    try:
        success = test_non_linearity_factory()
        results["Non-linearity Factory"] = "✓ PASSED" if success else "✗ FAILED"
    except Exception as e:
        print(f"\n✗ Non-linearity Factory FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["Non-linearity Factory"] = f"✗ FAILED: {e}"

    # Test 2: MLP Channel Mixer
    try:
        success = test_mlp_channel_mixer()
        results["MLP Channel Mixer"] = "✓ PASSED" if success else "✗ FAILED"
    except Exception as e:
        print(f"\n✗ MLP Channel Mixer FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["MLP Channel Mixer"] = f"✗ FAILED: {e}"

    # Test 3: Identity Channel Mixer
    try:
        success = test_identity_channel_mixer()
        results["Identity Channel Mixer"] = "✓ PASSED" if success else "✗ FAILED"
    except Exception as e:
        print(f"\n✗ Identity Channel Mixer FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["Identity Channel Mixer"] = f"✗ FAILED: {e}"

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"{name:30s}: {status}")

    # Check if all passed
    all_passed = all("PASSED" in status for status in results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
