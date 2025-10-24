"""Utility functions for linax."""

import equinox as eqx
import jax


def count_params(module) -> int:
    """Count the number of parameters in a module.

    Args:
        module:
            An equinox module (or any pytree containing arrays).

    Returns:
        Total number of parameters (sum of all array sizes).

    Example:
        ```python
        config = LinOSSConfig(...)
        model = config.build(key=key)
        total_params = count_params(model)
        print(f"Model has {total_params:,} parameters")
        ```
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    return sum(x.size for x in leaves)
