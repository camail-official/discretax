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

from linax import (
    LinOSS,
    LinOSSBackboneConfig,
    LinOSSConfig,
    LinOSSSequenceMixerConfig,
)


def example_1_default_config():
    """Example 1: Using default configuration."""
    print("=" * 80)
    print("Example 1: Default Configuration")
    print("=" * 80)

    # Create model with all defaults
    in_features = 32
    config = LinOSSConfig(
        hidden_dim=64,
        in_features=in_features,
    )
    key = jr.PRNGKey(0)

    model = LinOSS(cfg=config, key=key)

    print("\nModel Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Number of blocks: {config.backbone_config.num_blocks}")
    print(f"  Hidden dimension: {config.backbone_config.hidden_dim}")
    print(f"  Sequence mixer: {config.sequence_mixer_config.discretization}")
    print(f"  Damping: {config.sequence_mixer_config.damping}")

    print("\nModel Structure:")
    print(f"  Input features: {in_features}")
    print(f"  Output features: {model.out_features}")
    print(f"  Number of sequence mixers: {len(model.sequence_mixers)}")
    print(f"  Number of blocks: {len(model.backbone.blocks)}")

    # Create dummy input
    seq_len = 100
    x = jnp.ones((seq_len, in_features))

    # Initialize state
    state = eqx.nn.State(model)

    # Forward pass
    key = jr.PRNGKey(1)
    output, new_state = model(x, state, key)

    print("\nForward Pass:")
    print(f"  Input shape: {x.shape} (seq_len, in_features)")
    print(f"  Output shape: {output.shape} (out_features,)")
    print(f"  Output is pooled (classification mode): {config.backbone_config.classification}")

    print("\n✅ Example 1 complete!\n")
    return model, state


def example_2_custom_config():
    """Example 2: Custom configuration."""
    print("=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Define dimensions
    in_features = 64
    hidden_dim = 128

    # Create custom sequence mixer config
    # Note: hidden_dim will be set by LinOSSConfig.__post_init__
    sequence_mixer_config = LinOSSSequenceMixerConfig(
        name="my_custom_mixer",
        state_dim=128,  # State space dimension
        discretization="IM",  # Implicit midpoint (alternative to IMEX)
        damping=False,  # Disable damping
        r_min=0.8,  # Different eigenvalue radius
        theta_max=jnp.pi,
    )

    # Create custom backbone config
    # Note: hidden_dim will be set by LinOSSConfig.__post_init__
    backbone_config = LinOSSBackboneConfig(
        name="my_custom_backbone",
        num_blocks=8,  # More blocks for deeper model
        dropout_rate=0.15,  # Higher dropout
        classification=False,  # Regression mode (no pooling)
    )

    # Create model config
    # hidden_dim set here once, auto-propagates to mixer & backbone
    config = LinOSSConfig(
        name="my_custom_linoss",
        hidden_dim=hidden_dim,  # Set once - propagated automatically!
        in_features=in_features,
        sequence_mixer_config=sequence_mixer_config,
        backbone_config=backbone_config,
    )

    print("\nCustom Configuration:")
    print(f"  Model name: {config.name}")
    print(f"  Mixer discretization: {config.sequence_mixer_config.discretization}")
    print(f"  Mixer damping: {config.sequence_mixer_config.damping}")
    print(f"  Hidden dimension: {config.backbone_config.hidden_dim}")
    print(f"  Number of blocks: {config.backbone_config.num_blocks}")
    print(f"  Dropout rate: {config.backbone_config.dropout_rate}")
    print(f"  Classification mode: {config.backbone_config.classification}")

    # Initialize model
    key = jr.PRNGKey(42)
    model = LinOSS(cfg=config, key=key)

    print("\nModel Structure:")
    print(f"  Input features: {in_features}")
    print(f"  Output features: {model.out_features}")
    print(f"  Number of sequence mixers: {len(model.sequence_mixers)}")

    # Forward pass
    seq_len = 50
    x = jax.random.normal(jr.PRNGKey(2), (seq_len, in_features))
    state = eqx.nn.State(model)
    output, new_state = model(x, state, jr.PRNGKey(3))

    print("\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape} (seq_len, out_features) - regression mode")

    print("\n✅ Example 2 complete!\n")
    return model, state


def example_3_state_management():
    """Example 3: Proper state management across multiple forward passes."""
    print("=" * 80)
    print("Example 3: State Management")
    print("=" * 80)

    # Create model
    config = LinOSSConfig(in_features=16)
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

    config = LinOSSConfig(in_features=10)
    model = LinOSS(cfg=config, key=jr.PRNGKey(0))

    print("\nAccessing sequence mixers:")
    print(f"  model.sequence_mixers: list of {len(model.sequence_mixers)} mixers")
    print(f"  First mixer type: {type(model.sequence_mixers[0]).__name__}")
    print(f"  First mixer discretization: {model.sequence_mixers[0].discretization}")

    print("\nAccessing backbone:")
    print(f"  model.backbone.linear_encoder: {model.backbone.linear_encoder}")
    print(f"  model.backbone.linear_decoder: {model.backbone.linear_decoder}")
    print(f"  model.backbone.blocks: list of {len(model.backbone.blocks)} blocks")

    print("\nAccessing individual blocks:")
    first_block = model.backbone.blocks[0]
    print(f"  Block 0 sequence_mixer: {type(first_block.sequence_mixer).__name__}")
    print(f"  Block 0 GLU: {first_block.glu}")
    print(f"  Block 0 norm: {first_block.norm}")
    print(f"  Block 0 dropout: {first_block.drop}")

    print("\nVerifying sequence mixer assignment:")
    for i, block in enumerate(model.backbone.blocks):
        same_instance = block.sequence_mixer is model.sequence_mixers[i]
        print(f"  Block {i} uses model.sequence_mixers[{i}]: {same_instance}")

    print("\n✅ Example 4 complete!\n")


def example_5_partial_customization():
    """Example 5: Partial customization (mixing defaults with custom values)."""
    print("=" * 80)
    print("Example 5: Partial Customization")
    print("=" * 80)

    # Only customize the sequence mixer, use default backbone
    # Note: hidden_dim will be propagated from LinOSSConfig
    custom_mixer = LinOSSSequenceMixerConfig(
        discretization="IMEX",
        damping=True,
        state_dim=96,  # Custom state space dimension
    )

    # Backbone uses all defaults
    custom_backbone = LinOSSBackboneConfig(
        # All fields use defaults
    )

    # hidden_dim set at top level and propagated to both
    config = LinOSSConfig(
        hidden_dim=96,  # Set once - propagated automatically!
        in_features=24,
        sequence_mixer_config=custom_mixer,
        backbone_config=custom_backbone,
    )

    model = LinOSS(cfg=config, key=jr.PRNGKey(7))

    print("\nPartial Customization:")
    print(f"  Mixer hidden_dim (custom): {config.sequence_mixer_config.hidden_dim}")
    print(f"  Mixer discretization (custom): {config.sequence_mixer_config.discretization}")
    print(f"  Backbone num_blocks (default): {config.backbone_config.num_blocks}")
    print(f"  Backbone dropout_rate (default): {config.backbone_config.dropout_rate}")

    # Test forward pass
    x = jnp.ones((30, 24))
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
    example_5_partial_customization()

    print("=" * 80)
    print("All examples completed successfully! 🎉")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Use LinOSSConfig(in_features=N) for quick setup")
    print("  2. Customize LinOSSSequenceMixerConfig and LinOSSBackboneConfig for control")
    print("  3. Set hidden_dim once in LinOSSConfig - auto-propagates to components")
    print("  4. Always maintain state across forward passes")
    print("  5. Model owns sequence_mixers, which are passed to backbone blocks")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
