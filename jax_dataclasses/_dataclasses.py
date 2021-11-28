import dataclasses
from typing import Dict, List, Optional, Type, TypeVar

import jax

try:
    # Attempt to import flax for serialization. The exception handling lets us drop
    # flax from our dependencies.
    from flax import serialization
except ImportError:
    serialization = None  # type: ignore

from . import _copy_and_mutate

T = TypeVar("T")


FIELD_METADATA_STATIC_MARKER = "__jax_dataclasses_static_field__"


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


def static_field(*args, **kwargs):
    """Substitute for dataclasses.field, which also marks a field as static."""

    kwargs["metadata"] = kwargs.get("metadata", {})
    kwargs["metadata"][FIELD_METADATA_STATIC_MARKER] = True

    return dataclasses.field(*args, **kwargs)


def _register_pytree_dataclass(cls: Type[T]) -> Type[T]:
    """Register a dataclass as a flax-serializable pytree container.

    Args:
        cls (Type[T]): Dataclass to wrap.
    """

    assert dataclasses.is_dataclass(cls)

    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes.
    child_node_field_names: List[str] = []
    static_field_names: List[str] = []

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        if field.metadata.get(FIELD_METADATA_STATIC_MARKER, False):
            static_field_names.append(field.name)
        else:
            child_node_field_names.append(field.name)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        children = tuple(getattr(obj, key) for key in child_node_field_names)
        treedef = tuple(getattr(obj, key) for key in static_field_names)
        return children, treedef

    def _unflatten(treedef, children):
        return cls(
            **dict(zip(child_node_field_names, children)),
            **{key: tdef for key, tdef in zip(static_field_names, treedef)},
        )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`.
    if serialization is not None:

        def _to_state_dict(x: T):
            state_dict = {
                name: serialization.to_state_dict(getattr(x, name))
                for name in child_node_field_names
            }
            return state_dict

        def _from_state_dict(x: T, state: Dict):
            # Copy the state so we can pop the restored fields.
            state = state.copy()
            updates = {}
            for name in child_node_field_names:
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
            return dataclasses.replace(x, **updates)

        serialization.register_serialization_state(
            cls, _to_state_dict, _from_state_dict
        )

    # Custom frozen dataclass implementation
    cls.__mutability__ = _copy_and_mutate._Mutability.FROZEN  # type: ignore
    cls.__setattr__ = _copy_and_mutate._new_setattr  # type: ignore

    return cls
