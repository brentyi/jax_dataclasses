import contextlib
import dataclasses
import enum
from typing import Any, ContextManager, Set, TypeVar

from jax import numpy as jnp
from jax import tree_util
from jax.tree_util import default_registry

T = TypeVar("T")


class _Mutability(enum.Enum):
    FROZEN = enum.auto()
    MUTABLE = enum.auto()
    MUTABLE_NO_VALIDATION = enum.auto()


def _mark_mutable(obj: Any, mutable: _Mutability, visited: Set[Any]) -> None:
    """Recursively freeze or unfreeze dataclasses in a structure.
    Currently only supports tuples, lists, dictionaries, dataclasses."""

    # Skip objects we've already visited. This avoids redundancies when there are
    # identical branches in our pytree, but will also help prevent infinite looping from
    # cycles.
    if id(obj) in visited:
        return
    visited.add(id(obj))

    if dataclasses.is_dataclass(obj):
        object.__setattr__(obj, "__mutability__", mutable)

    flattened = default_registry.flatten_one_level(obj)
    if flattened is None:
        return
    for child in flattened[0]:
        _mark_mutable(child, mutable, visited)


def copy_and_mutate(obj: T, validate: bool = True) -> ContextManager[T]:
    """Context manager that copies a pytree and allows for temporary mutations to
    contained dataclasses. Optionally validates that treedefs, array shapes, and dtypes
    are not changed."""

    # Inner function helps with static typing!
    def _replace_context(obj: T):
        # Make a copy of the input object.
        obj_copy = tree_util.tree_map(lambda leaf: leaf, obj)

        # Mark it as mutable.
        _mark_mutable(
            obj_copy,
            mutable=(
                _Mutability.MUTABLE if validate else _Mutability.MUTABLE_NO_VALIDATION
            ),
            visited=set(),
        )

        # Yield.
        yield obj_copy

        # When done, mark as immutable again.
        _mark_mutable(
            obj_copy,
            mutable=_Mutability.FROZEN,
            visited=set(),
        )

    return contextlib.contextmanager(_replace_context)(obj)


def _unify_floats(dtype):
    if dtype == jnp.float64:
        return jnp.float32
    else:
        return dtype


def _new_setattr(self, name: str, value: Any):
    if self.__mutability__ == _Mutability.MUTABLE:
        # Validate changes.
        current_value = getattr(self, name)

        # Make sure tree structure is unchanged.
        assert tree_util.tree_structure(value) == tree_util.tree_structure(  # type: ignore
            current_value
        ), "Mismatched tree structure!"

        # Check leaf shapes.
        new_shapes = tuple(
            leaf.shape if hasattr(leaf, "shape") else tuple()
            for leaf in tree_util.tree_leaves(value)
        )
        cur_shapes = tuple(
            leaf.shape if hasattr(leaf, "shape") else tuple()
            for leaf in tree_util.tree_leaves(current_value)
        )
        assert (
            new_shapes == cur_shapes
        ), f"Shape error: new shapes {new_shapes} do not match original {cur_shapes}!"

        # Check leaf dtypes.
        new_dtypes = tuple(
            _unify_floats(leaf.dtype) if hasattr(leaf, "dtype") else type(leaf)
            for leaf in tree_util.tree_leaves(value)
        )
        cur_dtypes = tuple(
            _unify_floats(leaf.dtype) if hasattr(leaf, "dtype") else type(leaf)
            for leaf in tree_util.tree_leaves(current_value)
        )
        for new, cur in zip(new_dtypes, cur_dtypes):
            assert (
                new == cur or new in (int, float) or cur in (int, float)
            ), f"Type error: new dtypes {new_dtypes} do not match original {cur_dtypes}!"

        object.__setattr__(self, name, value)

    elif self.__mutability__ == _Mutability.MUTABLE_NO_VALIDATION:
        # Make changes without validation.
        object.__setattr__(self, name, value)

    elif self.__mutability__ == _Mutability.FROZEN:
        raise dataclasses.FrozenInstanceError(
            "Dataclass registered as pytree is immutable!"
        )

    else:
        assert False
