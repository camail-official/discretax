"""Test all SSM models (LinOSS, LRU, S5) to verify they work correctly."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from src.linax.encoder import LinearEncoderConfig
from src.linax.heads.classification import ClassificationHeadConfig
from src.linax.models import LinOSSConfig, LRUConfig, S5Config


def test_model(model_name: str, config, input_shape: tuple, num_classes: int):
    """Test a model with dummy data.

    Args:
        model_name: Name of the model being tested.
        config: Model configuration.
        input_shape: Shape of input data (batch_size, seq_len, in_features).
        num_classes: Number of output classes.
    """
    print(f"\n{'=' * 70}")
    print(f"Testing {model_name}")
    print(f"{'=' * 70}")

    # Create random key
    key = jr.PRNGKey(0)
    model_key, forward_key = jr.split(key, 2)

    # Build model
    print(f"\nBuilding {model_name}...")
    model = config.build(key=model_key)
    print(f"✓ {model_name} built successfully\n")
    print(model)

    # Create dummy input
    x = jr.normal(key, input_shape)
    print(f"\nInput shape: {x.shape}")
    print(f"Input stats - mean: {jnp.mean(x):.4f}, std: {jnp.std(x):.4f}")

    # Create state
    state = eqx.nn.State(model)

    # Forward pass
    print("\nRunning forward pass...")
    y, state = model(x, state, forward_key)

    print("✓ Forward pass completed")
    print(f"Output shape: {y.shape}")
    print(f"Output stats - mean: {jnp.mean(y):.4f}, std: {jnp.std(y):.4f}")

    # Verify output shape
    expected_shape = (num_classes,)
    assert y.shape == expected_shape, f"Output shape mismatch: {y.shape} != {expected_shape}"
    print(f"✓ Output shape correct: {y.shape}")

    # Verify output is valid
    assert not jnp.any(jnp.isnan(y)), "Output contains NaN!"
    print("✓ Output contains no NaN")

    assert not jnp.all(y == 0), "Output is all zeros!"
    print("✓ Output is not all zeros")

    print(f"\n✓ {model_name} test passed!")

    return model, y


def main():
    """Run tests for all models."""
    print("\n" + "=" * 70)
    print("TESTING ALL SSM MODELS")
    print("=" * 70)

    # Configuration
    in_features = 28 * 28  # MNIST-like
    seq_len = 1  # Single time step for now (treating flattened image as single timestep)
    hidden_dim = 64
    num_classes = 10
    num_blocks = 4

    # Input shape for SSM models: (seq_len, in_features) - no batch dimension
    input_shape = (seq_len, in_features)

    results = {}

    # Test 1: LinOSS Model
    try:
        print("\n" + "=" * 70)
        print("1. LinOSS Model")
        print("=" * 70)

        config = LinOSSConfig(
            num_blocks=num_blocks,
            encoder_config=LinearEncoderConfig(
                in_features=in_features,
                out_features=hidden_dim,
            ),
            head_config=ClassificationHeadConfig(out_features=num_classes),
        )

        model, output = test_model("LinOSS", config, input_shape, num_classes)
        results["LinOSS"] = "✓ PASSED"
    except Exception as e:
        print(f"\n✗ LinOSS FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["LinOSS"] = f"✗ FAILED: {e}"

    # Test 2: LRU Model
    try:
        print("\n" + "=" * 70)
        print("2. LRU Model")
        print("=" * 70)

        config = LRUConfig(
            num_blocks=num_blocks,
            encoder_config=LinearEncoderConfig(
                in_features=in_features,
                out_features=hidden_dim,
            ),
            head_config=ClassificationHeadConfig(out_features=num_classes),
        )

        model, output = test_model("LRU", config, input_shape, num_classes)
        results["LRU"] = "✓ PASSED"
    except Exception as e:
        print(f"\n✗ LRU FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["LRU"] = f"✗ FAILED: {e}"

    # Test 3: S5 Model
    try:
        print("\n" + "=" * 70)
        print("3. S5 Model")
        print("=" * 70)

        config = S5Config(
            num_blocks=num_blocks,
            encoder_config=LinearEncoderConfig(
                in_features=in_features,
                out_features=hidden_dim,
            ),
            head_config=ClassificationHeadConfig(out_features=num_classes),
        )

        model, output = test_model("S5", config, input_shape, num_classes)
        results["S5"] = "✓ PASSED"
    except Exception as e:
        print(f"\n✗ S5 FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["S5"] = f"✗ FAILED: {e}"

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"{name:20s}: {status}")

    # Check if all passed
    all_passed = all("PASSED" in status for status in results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL MODELS PASSED!")
    else:
        print("✗ SOME MODELS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
