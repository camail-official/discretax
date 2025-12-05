"""This module contains a mixin to load a class from a config.

This is useful for loading a class from a config file.
"""

import functools
import inspect
from typing import Annotated, Any, TypeVar, get_type_hints

ConfigVar = object()

T = TypeVar("T")
Cfg = Annotated[T, ConfigVar]


MODULE_REGISTRY = {}


def register(name: str):
    """Register a class in the registry."""

    def decorator(cls):
        MODULE_REGISTRY[name] = cls
        return cls

    return decorator


def build_from_config(cfg: Any) -> Any:
    """Recursively transforms a config dict into a generic Class Factory (partial)."""
    if isinstance(cfg, list):
        return [build_from_config(item) for item in cfg]

    if not isinstance(cfg, dict):
        return cfg

    if "name" in cfg and cfg["name"] in MODULE_REGISTRY:
        cls = MODULE_REGISTRY[cfg["name"]]

        processed_kwargs = {k: build_from_config(v) for k, v in cfg.items() if k != "name"}

        return functools.partial(cls, **processed_kwargs)

    return cfg


class PartialLoaderMixin:
    """Mixin to load a class from a config."""

    @classmethod
    def from_config(cls, cfg: dict | Any) -> functools.partial:
        """Returns a partial(cls, ...) with only the ConfigVars filled."""
        # Unwrap generic configs (like Hydra/Omegaconf) to dict
        if hasattr(cfg, "items"):
            cfg = dict(cfg)
        else:
            # Assume it's an object (dataclass/Namespace)
            cfg = vars(cfg)

        type_hints = get_type_hints(cls.__init__, include_extras=True)
        signature = inspect.signature(cls.__init__)

        config_args = {}

        for name, param in signature.parameters.items():
            if name == "self":
                continue
            # Look up the resolved hint
            hint = type_hints.get(name)

            # Check metadata
            if hasattr(hint, "__metadata__") and ConfigVar in hint.__metadata__:
                if name in cfg:
                    config_args[name] = cfg[name]
                elif param.default == inspect.Parameter.empty:
                    raise ValueError(f"Missing required config parameter: '{name}'")

        return functools.partial(cls, **config_args)
