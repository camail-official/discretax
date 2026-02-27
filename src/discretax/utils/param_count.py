"""Utility functions for parameter counting and printing."""

import equinox as eqx
import jax

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
}
_RESET = "\033[0m"


def _colorize(text: str, color: str | None) -> str:
    """Wrap text in ANSI color codes.

    Args:
        text: The string to colorize.
        color: Color name (one of red, green, yellow, blue, magenta, cyan). None for no color.

    Returns:
        The colorized string, or the original string if color is None or invalid.
    """
    if color is None or color not in _COLORS:
        return text
    return f"{_COLORS[color]}{text}{_RESET}"


def _print_tree(
    node: dict,
    prefix: str = "",
    path: tuple = (),
    current_depth: int = 1,
    depth: int | None = None,
    class_names: dict[tuple[str, ...], str] = {},
    color: str | None = None,
) -> None:
    """Recursively print a nested parameter tree.

    Args:
        node: Nested dict representing the parameter tree.
        prefix: Current line prefix (branch characters).
        path: Tuple of keys representing the current path in the tree.
        current_depth: Current recursion depth (1-indexed).
        depth: Maximum depth to print. None for unlimited.
        class_names: Mapping from path tuples to class name strings.
        color: Color name for class labels. None for no color.
    """
    if depth is not None and current_depth > depth:
        return
    children = node["__children__"] if "__children__" in node else node
    items = list(children.items())
    for i, (key, val) in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "
        extend = "    " if is_last else "│   "
        child_path = path + (key,)
        if "__leaf__" in val:
            if depth is None or current_depth <= depth:
                shape, size = val["__leaf__"]
                print(f"{prefix}{branch}{key}: {shape} ({size:,})")
        else:
            cls = class_names.get(child_path, "")
            cls_str = f" [{_colorize(cls, color)}]" if cls else ""
            print(f"{prefix}{branch}{key}/{cls_str} ({_sum_params(val):,})")
            _print_tree(
                val, prefix + extend, child_path, current_depth + 1, depth, class_names, color
            )


# Sum params in subtree
def _sum_params(node: dict) -> int:
    """Sum params in subtree."""
    if "__leaf__" in node:
        return node["__leaf__"][1]
    return sum(_sum_params(v) for v in node["__children__"].values())


def _get_key(p) -> str:
    if hasattr(p, "name"):
        return str(p.name)
    if hasattr(p, "key"):
        return str(p.key)
    return str(p.idx)


def count_params(module: eqx.Module) -> int:
    """Count the total number of trainable (inexact array) parameters in a module.

    Args:
        module: An Equinox module (or any JAX PyTree).

    Returns:
        Total number of floating-point parameters.

    Example:
        ```python
        model = LinOSS(hidden_dim=64, key=jax.random.PRNGKey(0))
        print(count_params(model))  # e.g. 123456
        ```
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    return sum(x.size for x in leaves)


def print_param_tree(
    model: eqx.Module, depth: int | None = None, color: str | None = None
) -> None:
    """Print a file-tree-like structure of an Equinox model's parameters.

    Args:
        model: The Equinox model (or any PyTree).
        depth: Maximum depth to print. None for unlimited.
        color: Color for class names (red, green, yellow, blue, magenta, cyan). None for no color.
    """
    params = eqx.filter(model, eqx.is_inexact_array)
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)

    # Build path -> class name mapping by walking the model
    class_names = {}

    def walk(obj, path=()):
        class_names[path] = type(obj).__name__
        if isinstance(obj, eqx.Module):
            for name, val in vars(obj).items():
                if isinstance(val, eqx.Module):
                    walk(val, path + (name,))
                elif isinstance(val, (list, tuple)):
                    for i, item in enumerate(val):
                        if isinstance(item, eqx.Module):
                            walk(item, path + (name, str(i)))

    walk(model)

    # Build nested dict tree
    tree = {}
    for path, leaf in leaves_with_path:
        keys = [_get_key(p) for p in path]
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(key, {"__children__": {}})["__children__"]
        node[keys[-1]] = {"__leaf__": (leaf.shape, leaf.size)}

    total = _sum_params({"__children__": tree})
    root_cls = class_names.get((), type(model).__name__)
    print(f"model/ [{_colorize(root_cls, color)}] ({total:,})")
    _print_tree({"__children__": tree}, path=(), depth=depth, class_names=class_names, color=color)
    print(f"\nTotal: {total:,}")


if __name__ == "__main__":
    import jax.random as jr

    from discretax.models import LinOSS

    model = LinOSS(
        hidden_dim=10,
        key=jr.PRNGKey(0),
    )
    print_param_tree(model, color="red")
