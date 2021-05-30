import dataclasses
from typing import Any, Dict, List, Optional, Type, TypeVar

import jax
from flax import serialization

T = TypeVar("T")


FIELD_METADATA_STATIC_MARKER = "__jax_dataclasses_static_field__"


def static_field(*args, **kwargs):
    kwargs["metadata"] = kwargs.get("metadata", {})
    kwargs["metadata"][FIELD_METADATA_STATIC_MARKER] = True

    return dataclasses.field(*args, **kwargs)


def dataclass(cls: Optional[Type] = None, **kwargs):
    def wrap(cls):
        return _register(dataclasses.dataclass(cls, **kwargs))

    if cls is None:
        return wrap
    else:
        return wrap(cls)


def _register(cls: Type[T]) -> Type[T]:
    """Register a dataclass as a flax-serializable PyTree.

    Args:
        cls (Type[T]): Dataclass to wrap.
    """

    assert dataclasses.is_dataclass(cls)

    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes
    child_node_field_names: List[str] = []
    static_fields: List[dataclasses.Field] = []
    for field in dataclasses.fields(cls):
        if FIELD_METADATA_STATIC_MARKER in field.metadata:
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
        return cls(
            **dict(zip(child_node_field_names, children)),
            **{field.name: tdef for field, tdef in zip(static_fields, treedef)},
        )

        # Alternative:
        #     return dataclasses.replace(
        #         cls.__new__(cls),
        #         **dict(zip(children_fields, children)),
        #         **dict(zip(static_fields_set, treedef)),
        #     )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`
    def _to_state_dict(x: T):
        state_dict = {
            name: serialization.to_state_dict(getattr(x, name))
            for name in child_node_field_names
        }
        return state_dict

    def _from_state_dict(x: T, state: Dict):
        state = state.copy()  # copy the state so we can pop the restored fields.
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

    cls.__is_update_buffer__ = False  # type: ignore

    # Make dataclass immutable after __init__ is called
    def _mark_immutable():
        original_init = cls.__init__ if hasattr(cls, "__init__") else None

        def new_init(self, *args, **kwargs):
            cls.__setattr__ = object.__setattr__
            if original_init is not None:
                original_init(self, *args, **kwargs)
            cls.__setattr__ = _new_setattr

        cls.__setattr__ = _new_setattr  # type: ignore
        cls.__init__ = new_init  # type: ignore

    _mark_immutable()

    return cls


def _new_setattr(self, name: str, value: Any):
    if self.__is_update_buffer__:
        current_value = getattr(self, name)
        assert jax.tree_structure(value) == jax.tree_structure(
            current_value
        ), "Mismatched tree structure!"

        new_shape_types = tuple(
            (leaf.shape, leaf.dtype) for leaf in jax.tree_leaves(value)
        )
        cur_shape_types = tuple(
            (leaf.shape, leaf.dtype) for leaf in jax.tree_leaves(current_value)
        )
        assert (
            new_shape_types == cur_shape_types
        ), f"Shape/type error: {new_shape_types} does not match {cur_shape_types}!"
        object.__setattr__(self, name, value)
    else:
        raise dataclasses.FrozenInstanceError(
            "Dataclass registered as PyTrees is immutable!"
        )
