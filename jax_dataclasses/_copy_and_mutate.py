import contextlib
import dataclasses
import enum
from typing import Any, ContextManager, Sequence, Set, TypeVar

from jax import numpy as jnp
from jax import tree_util
from jax._src.tree_util import _registry  # Dangerous!

T = TypeVar("T")


class _Mutability(enum.Enum):
    FROZEN = enum.auto()
    MUTABLE = enum.auto()
    MUTABLE_NO_VALIDATION = enum.auto()


def _tree_children(container: Any) -> Sequence[Any]:
    """Grab child nodes of a pytree. This would ideally be implemented using the pytree
    registry."""

    registry_entry = _registry.get(type(container))
    if registry_entry is not None:
        children, _metadata = registry_entry.to_iter(container)
        return list(children)
    return []


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

    for child in _tree_children(obj):
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
            mutable=_Mutability.MUTABLE
            if validate
            else _Mutability.MUTABLE_NO_VALIDATION,
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
        assert tree_util.tree_structure(value) == tree_util.tree_structure(
            current_value
        ), "Mismatched tree structure!"

        # Check leaf shapes.
        new_shapes = tuple(
            leaf.shape
            for leaf in tree_util.tree_leaves(value)
            if hasattr(leaf, "shape")
        )
        cur_shapes = tuple(
            leaf.shape
            for leaf in tree_util.tree_leaves(current_value)
            if hasattr(leaf, "shape")
        )
        assert (
            new_shapes == cur_shapes
        ), f"Shape error: {new_shapes} does not match {cur_shapes}!"

        # Check leaf dtypes.
        new_dtypes = tuple(
            _unify_floats(leaf.dtype)
            for leaf in tree_util.tree_leaves(value)
            if hasattr(leaf, "dtype")
        )
        cur_dtypes = tuple(
            _unify_floats(leaf.dtype)
            for leaf in tree_util.tree_leaves(current_value)
            if hasattr(leaf, "dtype")
        )
        assert (
            new_dtypes == cur_dtypes
        ), f"Type error: {new_dtypes} does not match {cur_dtypes}!"

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
