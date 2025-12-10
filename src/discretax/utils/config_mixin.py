"""Config mixin."""

import warnings
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, Self, TypeVar, Union

ConfigVar = object()

P = ParamSpec("P")
T = TypeVar("T")


class Partial(Generic[P, T]):
    """Wrapper holding a class + config kwargs, resolved with runtime kwargs.

    Runtime kwargs can override config kwargs (with a warning).
    Uses ParamSpec to preserve the __init__ signature for IDE autocomplete.
    """

    cls: Callable[P, T]
    kwargs: dict[str, Any]

    def __init__(self, cls: Callable[P, T], **kwargs: Any):
        self.cls = cls
        self.kwargs = kwargs

    def resolve(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Instantiate the class with config + runtime kwargs.

        Args:
            *args: Positional args for __init__
            **kwargs: Keyword args for __init__. Config params can be
                overridden (issues warning).

        Returns:
            Fully instantiated object of type T.
        """
        # Check for overrides and warn
        overridden = set(self.kwargs.keys()) & set(kwargs.keys())
        for key in overridden:
            warnings.warn(
                f"Config param '{key}' is being overridden: {self.kwargs[key]} -> {kwargs[key]}",
                stacklevel=2,
            )

        # Merge: runtime kwargs override config kwargs
        merged = {**self.kwargs, **kwargs}
        return self.cls(*args, **merged)

    def __repr__(self) -> str:
        """Representation of the Partial object."""
        cls_name = getattr(self.cls, "__name__", str(self.cls))
        return f"Partial({cls_name}, {self.kwargs})"


# Type alias for parameters that can be either a Partial or already resolved
Resolvable = Union[Partial[..., T], T]


class PartialModule:
    """Mixin for classes that can be partially initialized.

    Subclasses get a .resolve() method that returns self (already resolved).
    """

    def resolve(self, *args: Any, **kwargs: Any) -> Self:
        """If called on an instance, return self (already resolved)."""
        return self
