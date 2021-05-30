import contextlib
import dataclasses
from typing import Any, ContextManager, TypeVar

import jax

T = TypeVar("T")


def _mark_mutable(obj: Any, mutable: bool) -> None:
    """Recursively freeze or unfreeze dataclasses in a structure.
    Currently only supports tuples, lists, dictionaries, dataclasses."""

    if isinstance(obj, (tuple, list)):
        for child in obj:
            _mark_mutable(child, mutable)
    elif isinstance(obj, dict):
        for child in obj.values():
            _mark_mutable(child, mutable)
    elif dataclasses.is_dataclass(obj):
        object.__setattr__(obj, "__is_update_buffer__", mutable)
        for child in vars(obj).values():
            _mark_mutable(child, mutable)


def copy_and_mutate(obj: T) -> ContextManager[T]:
    """Context manager that copies a PyTree and allows for temporary mutations to
    contained dataclasses. Also validates that treedefs, array shapes, etc are
    not changed."""

    # Inner function helps with static typing
    def _replace_context(obj: T):
        # Make a copy of the input object
        obj_copy = jax.tree_map(lambda leaf: leaf, obj)

        # Mark it as mutable
        _mark_mutable(obj_copy, mutable=True)

        # Yield
        yield obj_copy

        # When done, mark as immutable again
        _mark_mutable(obj_copy, mutable=False)

    return contextlib.contextmanager(_replace_context)(obj)
