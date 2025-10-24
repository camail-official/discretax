"""Basic usage examples and tests for the LinOSS model.

This script demonstrates:
1. Using default configuration
2. Creating custom configurations
3. Forward pass with state management
4. Accessing model components
5. Custom dimensions
6. Using high-level vs low-level configs
7. Model summary with __repr__
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from linax.architecture.blocks.linoss import LinOSSBlockConfig
from linax.architecture.encoder.linear import LinearEncoderConfig
from linax.architecture.heads.classification import ClassificationHeadConfig
from linax.architecture.models.linoss import LinOSSConfig
from linax.architecture.models.ssm import SSMConfig
from linax.architecture.sequence_mixers.linoss import LinOSSSequenceMixerConfig


def example_1_default_config():
    """Example 1: Using default configuration."""
    print("=" * 80)
    print("Example 1: Default Configuration")
    print("=" * 80)

    # Create model with standard MNIST-like settings
    config = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=784, out_features=20),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=20),
        block_config=LinOSSBlockConfig(drop_rate=0.1),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    key = jr.PRNGKey(0)

    model = config.build(key=key)

    print("\nModel Configuration:")
    print(f"  Input features: {config.encoder_config.in_features}")
    print(f"  Hidden dimension: {config.encoder_config.out_features}")
    print(f"  Output features: {config.head_config.out_features}")
    print(f"  Number of blocks: {config.num_blocks}")
    print(f"  Dropout rate: {config.block_config.drop_rate}")

    print("\nModel Structure:")
    print(f"  Encoder: {type(model.encoder).__name__}")
    print(f"  Number of blocks: {len(model.blocks)}")
    print(f"  Head: {type(model.head).__name__}")

    # Create dummy input
    seq_len = 100
    x = jnp.ones((seq_len, config.encoder_config.in_features))

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
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=64, out_features=128),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=128),
        block_config=LinOSSBlockConfig(drop_rate=0.1),
        head_config=ClassificationHeadConfig(out_features=20),
    )

    # Initialize model
    key = jr.PRNGKey(42)
    model = config.build(key=key)

    print("\nCustom Configuration:")
    print(f"  Input features: {config.encoder_config.in_features}")
    print(f"  Hidden dimension: {config.encoder_config.out_features}")
    print(f"  Output features: {config.head_config.out_features}")

    print("\nModel Structure:")
    print(f"  Encoder: {type(model.encoder).__name__}")
    print(f"  Number of blocks: {len(model.blocks)}")
    print(f"  Head: {type(model.head).__name__}")

    # Forward pass
    seq_len = 50
    x = jax.random.normal(jr.PRNGKey(2), (seq_len, config.encoder_config.in_features))
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
    config = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=16, out_features=32),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=32),
        block_config=LinOSSBlockConfig(),
        head_config=ClassificationHeadConfig(out_features=5),
    )
    model = config.build(key=jr.PRNGKey(0))

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

    config = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=10, out_features=32),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=32),
        block_config=LinOSSBlockConfig(),
        head_config=ClassificationHeadConfig(out_features=5),
    )
    model = config.build(key=jr.PRNGKey(0))

    print("\nAccessing encoder:")
    print(f"  model.encoder: {type(model.encoder).__name__}")

    print("\nAccessing blocks:")
    print(f"  model.blocks: list of {len(model.blocks)} blocks")
    print(f"  First block type: {type(model.blocks[0]).__name__}")

    print("\nAccessing head:")
    print(f"  model.head: {type(model.head).__name__}")

    print("\nAccessing individual block components:")
    first_block = model.blocks[0]
    print(f"  Block 0 sequence_mixer: {type(first_block.sequence_mixer).__name__}")
    print(f"  Block 0 mlp: {type(first_block.mlp).__name__}")
    print(f"  Block 0 norm: {type(first_block.norm).__name__}")
    print(f"  Block 0 drop: {type(first_block.drop).__name__}")

    print("\n✅ Example 4 complete!\n")


def example_5_custom_dimensions():
    """Example 5: Custom dimensions for different use cases."""
    print("=" * 80)
    print("Example 5: Custom Dimensions")
    print("=" * 80)

    # Create config for a specific use case
    config = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=28, out_features=64),  # MNIST-like input
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=64),
        block_config=LinOSSBlockConfig(),
        head_config=ClassificationHeadConfig(out_features=10),  # 10 classes
    )

    model = config.build(key=jr.PRNGKey(7))

    print("\nCustom Dimensions:")
    print(f"  Input features: {config.encoder_config.in_features}")
    print(f"  Hidden dimension: {config.encoder_config.out_features}")
    print(f"  Output features: {config.head_config.out_features}")

    # Test forward pass
    x = jnp.ones((30, config.encoder_config.in_features))
    state = eqx.nn.State(model)
    output, state = model(x, state, jr.PRNGKey(8))

    print(f"\nForward pass successful: {output.shape}")
    print("\n✅ Example 5 complete!\n")


def example_6_high_vs_low_level_config():
    """Example 6: High-level vs Low-level configuration."""
    print("=" * 80)
    print("Example 6: High-level vs Low-level Config")
    print("=" * 80)

    # High-level config (simplified, user-friendly)
    print("\n📝 High-level Config (LinOSSConfig):")
    print("   - Simple hyperparameters")
    print("   - Automatic component composition")
    print("   - Best for quick prototyping")
    print("   - LinOSSConfig inherits from SSMConfig\n")

    high_level_cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=32, out_features=64),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=64),
        block_config=LinOSSBlockConfig(drop_rate=0.1),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    model_high = high_level_cfg.build(key=jr.PRNGKey(0))
    print(f"  Created model with {len(model_high.blocks)} blocks")
    print(f"  Encoder config: {type(high_level_cfg.encoder_config).__name__}")
    print(f"  Block config: {type(high_level_cfg.block_configs[0]).__name__}")
    print(f"  Head config: {type(high_level_cfg.head_config).__name__}")

    # Low-level config (component-based, fine-grained control)
    print("\n🔧 Low-level Config (SSMConfig):")
    print("   - Component-based configuration")
    print("   - Fine-grained control")
    print("   - Best for custom architectures\n")

    ssm_cfg = SSMConfig(
        encoder_config=LinearEncoderConfig(in_features=32, out_features=64),
        sequence_mixer_configs=[LinOSSSequenceMixerConfig(state_dim=64)] * 4,
        block_configs=[LinOSSBlockConfig(drop_rate=0.1)] * 4,
        head_config=ClassificationHeadConfig(out_features=10),
    )
    model_low = ssm_cfg.build(key=jr.PRNGKey(0))
    print(f"  Created model with {len(model_low.blocks)} blocks")
    print(f"  Encoder config: {type(ssm_cfg.encoder_config).__name__}")
    print(f"  Block config: {type(ssm_cfg.block_configs[0]).__name__}")
    print(f"  Head config: {type(ssm_cfg.head_config).__name__}")

    # Both produce equivalent models
    print("\n✅ Both approaches produce equivalent SSM models!")
    print("\n✅ Example 6 complete!\n")


def example_7_model_summary():
    """Example 7: Model summary with __repr__."""
    print("=" * 80)
    print("Example 7: Model Summary")
    print("=" * 80)

    print("\n📊 General SSM Summary:")
    # Create a general SSM
    from linax.architecture.blocks.linoss import LinOSSBlockConfig
    from linax.architecture.encoder import LinearEncoderConfig
    from linax.architecture.heads.classification import ClassificationHeadConfig
    from linax.architecture.sequence_mixers.linoss import LinOSSSequenceMixerConfig

    ssm_config = SSMConfig(
        encoder_config=LinearEncoderConfig(in_features=100, out_features=32),
        sequence_mixer_configs=[LinOSSSequenceMixerConfig(state_dim=32)] * 2,
        block_configs=[LinOSSBlockConfig(drop_rate=0.2)] * 2,
        head_config=ClassificationHeadConfig(out_features=5),
    )
    ssm_model = ssm_config.build(key=jr.PRNGKey(42))
    print(ssm_model)

    print("\n📊 LinOSS Model Summary (with extra details):")
    # Create a LinOSS model
    linoss_config = LinOSSConfig(
        num_blocks=2,
        encoder_config=LinearEncoderConfig(in_features=28, out_features=32),
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=32),
        block_config=LinOSSBlockConfig(drop_rate=0.15),
        head_config=ClassificationHeadConfig(out_features=10),
    )
    linoss_model = linoss_config.build(key=jr.PRNGKey(42))
    print(linoss_model)

    print("\n✅ Example 7 complete!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LinOSS Model Usage Examples and Tests")
    print("=" * 80 + "\n")

    # Run all examples
    example_1_default_config()
    example_2_custom_config()
    example_3_state_management()
    example_4_accessing_components()
    example_5_custom_dimensions()
    example_6_high_vs_low_level_config()
    example_7_model_summary()

    print("=" * 80)
    print("All examples completed successfully! 🎉")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. LinOSSConfig requires: in_features, hidden_dim, out_features")
    print("  2. Optional parameters: num_blocks (default 4), drop_rate (default 0.1)")
    print("  3. Build models with config.build(key=...)")
    print("  4. Always maintain state across forward passes")
    print("  5. Model structure: encoder → blocks → head")
    print("  6. Each LinOSSBlock contains: sequence_mixer + mlp + norm + dropout")
    print("  7. High-level configs (LinOSSConfig) for simplicity")
    print("  8. Low-level configs (SSMConfig) for custom architectures")
    print("  9. Use print(model) to see detailed model summary")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
