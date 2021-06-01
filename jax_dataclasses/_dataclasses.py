import dataclasses
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar

import jax
from flax import serialization

from . import _copy_and_mutate

T = TypeVar("T")


FIELD_METADATA_STATIC_MARKER = "__jax_dataclasses_static_field__"


if TYPE_CHECKING:
    # Treat our JAX field and dataclass functions as their counterparts from the
    # standard dataclasses library during static analysis
    #
    # Tools like via mypy, jedi, etc generally rely on a lot of special, hardcoded
    # behavior for the standard dataclasses library; this lets us take advantage of all
    # of it.
    #
    # Note that mypy will not follow aliases, so `from dataclasses import dataclass` is
    # preferred over `dataclass = dataclasses.dataclass`.
    #
    # For the future, dataclass transforms may also be worth looking into:
    # https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
    from dataclasses import dataclass
    from dataclasses import field as static_field
else:

    def static_field(*args, **kwargs):
        """Substitute for dataclasses.field, which also marks a field as static."""

        kwargs["metadata"] = kwargs.get("metadata", {})
        kwargs["metadata"][FIELD_METADATA_STATIC_MARKER] = True

        return dataclasses.field(*args, **kwargs)

    def dataclass(cls: Optional[Type] = None, **kwargs):
        """Substitute for dataclasses.dataclass, which also registers dataclasses as
        PyTrees."""

        def wrap(cls):
            return _register(dataclasses.dataclass(cls, **kwargs))

        if "frozen" in kwargs:
            assert kwargs["frozen"] is True, "Pytree dataclasses can only be frozen!"

        if cls is None:
            return wrap
        else:
            return wrap(cls)


def _register(cls: Type[T]) -> Type[T]:
    """Register a dataclass as a flax-serializable pytree.

    Args:
        cls (Type[T]): Dataclass to wrap.
    """

    assert dataclasses.is_dataclass(cls)

    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes.
    child_node_field_names: List[str] = []
    static_fields: List[dataclasses.Field] = []
    for field in dataclasses.fields(cls):
        if (
            FIELD_METADATA_STATIC_MARKER in field.metadata
            and field.metadata[FIELD_METADATA_STATIC_MARKER]
        ):
            static_fields.append(field)
        else:
            child_node_field_names.append(field.name)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        children = tuple(getattr(obj, key) for key in child_node_field_names)
        treedef = tuple(getattr(obj, field.name) for field in static_fields)
        return children, treedef

    def _unflatten(treedef, children):
        static_field_names = [field.name for field in static_fields]
        return dataclasses.replace(
            cls.__new__(cls),
            **dict(zip(child_node_field_names, children)),
            **dict(zip(static_field_names, treedef)),
        )
        # Alternative:
        # return cls(
        #     **dict(zip(child_node_field_names, children)),
        #     **{field.name: tdef for field, tdef in zip(static_fields, treedef)},
        # )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`.
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

    serialization.register_serialization_state(cls, _to_state_dict, _from_state_dict)

    cls.__mutability__ = _copy_and_mutate._Mutability.FROZEN  # type: ignore

    # Make dataclass immutable after __init__ is called.
    def _mark_immutable():
        original_init = cls.__init__ if hasattr(cls, "__init__") else None

        def new_init(self, *args, **kwargs):
            # Allow mutations in __init__.
            cls.__setattr__ = object.__setattr__
            if original_init is not None:
                original_init(self, *args, **kwargs)
            cls.__setattr__ = _copy_and_mutate._new_setattr

        cls.__setattr__ = _copy_and_mutate._new_setattr  # type: ignore
        cls.__init__ = new_init  # type: ignore

    _mark_immutable()

    return cls
