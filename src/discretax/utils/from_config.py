"""Initialize a model from a configuration dictionary."""

import importlib
import pkgutil
from typing import Any

import jax.random as jr

import discretax
from discretax.utils import print_param_tree
from discretax.utils.config_mixin import Partial, PartialModule


def _resolve_target(target: str) -> type:
    """Resolve a target string to a class, searching discretax submodules if needed."""
    if "." in target:
        module_path, class_name = target.rsplit(".", 1)
        target_module = importlib.import_module(module_path)
        return getattr(target_module, class_name)
    # No dot - search through discretax submodules
    for importer, modname, ispkg in pkgutil.walk_packages(discretax.__path__, "discretax."):
        try:
            module = importlib.import_module(modname)
            if hasattr(module, target):
                return getattr(module, target)
        except ImportError:
            continue
    raise ImportError(f"Could not find class '{target}' in discretax")


def build_from_dict(cfg: dict[str, Any]) -> PartialModule:
    """Build model from cfg.

    Args:
        cfg: Configuration dictionary.

    Returns:
        PartialModule: Partial module.
    """
    if isinstance(cfg, dict) and "target" in cfg:
        target = cfg["target"]
        kwargs = {k: build_from_dict(v) for k, v in cfg.items() if k != "target"}
        target_cls = _resolve_target(target)
        return Partial(target_cls, **kwargs)
    elif isinstance(cfg, dict):
        return {k: build_from_dict(v) for k, v in cfg.items()}
    else:
        return cfg


if __name__ == "__main__":
    cfg = {
        "target": "StandardBlock",
        "sequence_mixer": {
            "target": "LinOSSSequenceMixer",
            "state_dim": 20,
        },
        "channel_mixer": {
            "target": "GLU",
            "use_bias": True,
        },
        "drop_rate": 1.0,
    }
    model_key = jr.key(0)
    model = build_from_dict(cfg)
    print(model)
    print_param_tree(model.resolve(in_features=10, key=model_key))
