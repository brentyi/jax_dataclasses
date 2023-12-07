from __future__ import annotations

import dataclasses
import functools
from typing import Dict, List, Optional, Type, TypeVar

from jax import tree_util
from typing_extensions import Annotated, get_type_hints

from ._get_type_hints import get_type_hints_partial

try:
    # Attempt to import flax for serialization. The exception handling lets us drop
    # flax from our dependencies.
    from flax import serialization
except ImportError:
    serialization = None  # type: ignore

from . import _copy_and_mutate

T = TypeVar("T")


JDC_STATIC_MARKER = "__jax_dataclasses_static_field__"


# Stolen from here: https://github.com/google/jax/issues/10476
InnerT = TypeVar("InnerT")
Static = Annotated[InnerT, JDC_STATIC_MARKER]
"""Annotates a type as static in the sense of JAX; in a pytree, fields marked as such
should be hashable and are treated as part of the treedef and not as a child node."""


def pytree_dataclass(cls: Optional[Type] = None, **kwargs):
    """Substitute for dataclasses.dataclass, which also registers dataclasses as
    PyTrees."""

    def wrap(cls):
        return _register_pytree_dataclass(dataclasses.dataclass(cls, **kwargs))

    if "frozen" in kwargs:
        assert kwargs["frozen"] is True, "Pytree dataclasses can only be frozen!"
    kwargs["frozen"] = True

    if cls is None:
        return wrap
    else:
        return wrap(cls)


def deprecated_static_field(*args, **kwargs):
    """Deprecated, prefer `Static[]` on the type annotation instead."""

    kwargs["metadata"] = kwargs.get("metadata", {})
    kwargs["metadata"][JDC_STATIC_MARKER] = True

    return dataclasses.field(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    child_node_field_names: List[str]
    static_field_names: List[str]


def _register_pytree_dataclass(cls: Type[T]) -> Type[T]:
    """Register a dataclass as a flax-serializable pytree container."""

    assert dataclasses.is_dataclass(cls)

    @functools.lru_cache(maxsize=1)
    def get_field_info() -> FieldInfo:
        # Determine which fields are static and part of the treedef, and which should be
        # registered as child nodes.
        child_node_field_names: List[str] = []
        static_field_names: List[str] = []

        # We don't directly use field.type for postponed evaluation; we want to make sure
        # that our types are interpreted as proper types and not as (string) forward
        # references.
        #
        # Note that there are ocassionally situations where the @jdc.pytree_dataclass
        # decorator is called before a referenced type is defined; to suppress this error,
        # we resolve missing names to our subscriptible placeholder object.

        try:
            type_from_name = get_type_hints(cls, include_extras=True)  # type: ignore
        except Exception:
            # Try again, but suppress errors from unresolvable forward
            # references. This should be rare.
            type_from_name = get_type_hints_partial(cls, include_extras=True)  # type: ignore

        for field in dataclasses.fields(cls):
            if not field.init:
                continue

            field_type = type_from_name[field.name]

            # Two ways to mark a field as static: either via the Static[] type or
            # jdc.static_field().
            if (
                hasattr(field_type, "__metadata__")
                and JDC_STATIC_MARKER in field_type.__metadata__
            ):
                static_field_names.append(field.name)
                continue
            if field.metadata.get(JDC_STATIC_MARKER, False):
                static_field_names.append(field.name)
                continue

            child_node_field_names.append(field.name)
        return FieldInfo(child_node_field_names, static_field_names)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        field_info = get_field_info()
        children = tuple(getattr(obj, key) for key in field_info.child_node_field_names)
        treedef = tuple(getattr(obj, key) for key in field_info.static_field_names)
        return children, treedef

    def _unflatten(treedef, children):
        field_info = get_field_info()
        return cls(
            **dict(zip(field_info.child_node_field_names, children)),
            **{key: tdef for key, tdef in zip(field_info.static_field_names, treedef)},
        )

    tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`.
    if serialization is not None:

        def _to_state_dict(x: T):
            field_info = get_field_info()
            state_dict = {
                name: serialization.to_state_dict(getattr(x, name))
                for name in field_info.child_node_field_names
            }
            return state_dict

        def _from_state_dict(x: T, state: Dict):
            # Copy the state so we can pop the restored fields.
            field_info = get_field_info()
            state = state.copy()
            updates = {}
            for name in field_info.child_node_field_names:
                if name not in state:
                    raise ValueError(
                        f"Missing field {name} in state dict while restoring"
                        f" an instance of {cls.__name__}"
                    )
                value = getattr(x, name)
                value_state = state.pop(name)
                updates[name] = serialization.from_state_dict(value, value_state)
            if state:
                names = ",".join(state.keys())
                raise ValueError(
                    f'Unknown field(s) "{names}" in state dict while'
                    f" restoring an instance of {cls.__name__}"
                )

            return dataclasses.replace(x, **updates)  # type: ignore

        serialization.register_serialization_state(
            cls, _to_state_dict, _from_state_dict
        )

    # Custom frozen dataclass implementation
    cls.__mutability__ = _copy_and_mutate._Mutability.FROZEN  # type: ignore
    cls.__setattr__ = _copy_and_mutate._new_setattr  # type: ignore

    return cls  # type: ignore
