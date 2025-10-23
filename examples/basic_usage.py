"""Basic usage examples for the LinOSS model.

This script demonstrates:
1. Using default configuration
2. Creating custom configurations
3. Forward pass with state management
4. Accessing model components
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from linax.architecture.models.linoss import (
    LinOSS,
    LinOSSConfig,
)


def example_1_default_config():
    """Example 1: Using default configuration."""
    print("=" * 80)
    print("Example 1: Default Configuration")
    print("=" * 80)

    # Create model with all defaults
    config = LinOSSConfig()
    key = jr.PRNGKey(0)

    model = LinOSS(cfg=config, key=key)

    print("\nModel Configuration:")
    print(f"  Input features: {config.in_features}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Output features: {config.out_features}")

    print("\nModel Structure:")
    print(f"  Number of sequence mixers: {len(model.sequence_mixers)}")
    print(f"  Number of blocks: {len(model.blocks)}")

    # Create dummy input
    seq_len = 100
    x = jnp.ones((seq_len, config.in_features))

    # Initialize state
    state = eqx.nn.State(model)

    # Forward pass
    key = jr.PRNGKey(1)
    output, new_state = model(x, state, key)

    print("\nForward Pass:")
    print(f"  Input shape: {x.shape} (seq_len, in_features)")
    print(f"  Output shape: {output.shape} (out_features,)")

    print("\n✅ Example 1 complete!\n")
    return model, state


def example_2_custom_config():
    """Example 2: Custom configuration."""
    print("=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Create custom config with specific values
    config = LinOSSConfig(
        in_features=64,
        hidden_dim=128,
        out_features=20,
    )

    # Initialize model
    key = jr.PRNGKey(42)
    model = LinOSS(cfg=config, key=key)

    print("\nCustom Configuration:")
    print(f"  Input features: {config.in_features}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Output features: {config.out_features}")

    print("\nModel Structure:")
    print(f"  Number of sequence mixers: {len(model.sequence_mixers)}")
    print(f"  Number of blocks: {len(model.blocks)}")

    # Forward pass
    seq_len = 50
    x = jax.random.normal(jr.PRNGKey(2), (seq_len, config.in_features))
    state = eqx.nn.State(model)
    output, new_state = model(x, state, jr.PRNGKey(3))

    print("\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    print("\n✅ Example 2 complete!\n")
    return model, state


def example_3_state_management():
    """Example 3: Proper state management across multiple forward passes."""
    print("=" * 80)
    print("Example 3: State Management")
    print("=" * 80)

    # Create model
    config = LinOSSConfig(in_features=16, out_features=5)
    model = LinOSS(cfg=config, key=jr.PRNGKey(0))

    # Initialize state
    state = eqx.nn.State(model)

    print("\nProcessing multiple sequences:")

    # Process multiple sequences, maintaining state
    for i in range(3):
        # Create input
        x = jax.random.normal(jr.PRNGKey(i), (20, 16))

        # Forward pass - always capture the new state!
        output, state = model(x, state, jr.PRNGKey(100 + i))

        print(f"  Sequence {i + 1}: Input {x.shape} → Output {output.shape}")

    print("\n⚠️  Important: Always capture and pass forward the returned state!")
    print("   Correct:   output, state = model(x, state, key)")
    print("   Incorrect: output, _ = model(x, state, key)  # Loses state updates!")

    print("\n✅ Example 3 complete!\n")


def example_4_accessing_components():
    """Example 4: Accessing model components."""
    print("=" * 80)
    print("Example 4: Accessing Model Components")
    print("=" * 80)

    config = LinOSSConfig(in_features=10, out_features=5)
    model = LinOSS(cfg=config, key=jr.PRNGKey(0))

    print("\nAccessing sequence mixers:")
    print(f"  model.sequence_mixers: list of {len(model.sequence_mixers)} mixers")
    print(f"  First mixer type: {type(model.sequence_mixers[0]).__name__}")

    print("\nAccessing encoder:")
    print(f"  model.encoder: {type(model.encoder).__name__}")

    print("\nAccessing blocks:")
    print(f"  model.blocks: list of {len(model.blocks)} blocks")

    print("\nAccessing head:")
    print(f"  model.head: {type(model.head).__name__}")

    print("\nAccessing individual blocks:")
    first_block = model.blocks[0]
    print(f"  Block 0 sequence_mixer: {type(first_block.sequence_mixer).__name__}")
    print(f"  Block 0 GLU: {first_block.mlp}")
    print(f"  Block 0 norm: {first_block.norm}")
    print(f"  Block 0 dropout: {first_block.drop}")

    print("\nVerifying sequence mixer assignment:")
    for i, block in enumerate(model.blocks):
        same_instance = block.sequence_mixer is model.sequence_mixers[i]
        print(f"  Block {i} uses model.sequence_mixers[{i}]: {same_instance}")

    print("\n✅ Example 4 complete!\n")


def example_5_custom_dimensions():
    """Example 5: Custom dimensions for different use cases."""
    print("=" * 80)
    print("Example 5: Custom Dimensions")
    print("=" * 80)

    # Create config for a specific use case
    config = LinOSSConfig(
        in_features=28,  # MNIST-like input
        hidden_dim=64,  # Hidden representation
        out_features=10,  # 10 classes
    )

    model = LinOSS(cfg=config, key=jr.PRNGKey(7))

    print("\nCustom Dimensions:")
    print(f"  Input features: {config.in_features}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Output features: {config.out_features}")

    # Test forward pass
    x = jnp.ones((30, config.in_features))
    state = eqx.nn.State(model)
    output, state = model(x, state, jr.PRNGKey(8))

    print(f"\nForward pass successful: {output.shape}")
    print("\n✅ Example 5 complete!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LinOSS Model Usage Examples")
    print("=" * 80 + "\n")

    # Run all examples
    example_1_default_config()
    example_2_custom_config()
    example_3_state_management()
    example_4_accessing_components()
    example_5_custom_dimensions()

    print("=" * 80)
    print("All examples completed successfully! 🎉")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Use LinOSSConfig() for quick setup with defaults")
    print("  2. Customize dimensions: in_features, hidden_dim, out_features")
    print("  3. Always maintain state across forward passes")
    print("  4. Model structure: encoder → blocks → head")
    print("  5. Model owns sequence_mixers, which are passed to blocks")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
