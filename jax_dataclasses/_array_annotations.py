import dataclasses
from typing import Any, List, Optional, Tuple

from jax import numpy as jnp
from typing_extensions import get_type_hints


def _is_shape(shape: Any) -> bool:
    """Returns `True` is `shape` is of type `Tuple[int, ...]`."""
    return isinstance(shape, tuple) and all(map(lambda x: isinstance(x, int), shape))


# Some dtype superclasses that result in a warning when we ttempt a jnp.dtype() on them.
_dtype_set = {jnp.integer, jnp.signedinteger, jnp.floating, jnp.inexact}


def _is_dtype(dtype: Any) -> bool:
    """Returns `True` is `dtype` is a valid datatype."""

    if dtype in _dtype_set:
        return True
    try:
        jnp.dtype(dtype)
        return True
    except TypeError:
        return False


class ArrayAnnotationMixin:
    """Base class for dataclasses containing arrays that are annotated with expected
    shapes or types that can be checked at runtime.

    Runs input validation on instantiation and provides a single helper,
    `validate_and_get_batch_axes()`, that returns common batch axes.

    First, we import `Annotated`:

        # Python <=3.8
        from typing_extensions import Annotated

        # Python 3.9
        from typing import Annotated

    Example of an annotated fields that must have shape (*, 50, 150, 3) and (*, 10),
    where the batch axes must be shared:

        image: Annotated[jnp.ndarray, (50, 150, 3)]
        label: Annotated[jnp.ndarray, (10,)]

    Fields that must be floats and integers respectively:

        image: Annotated[jnp.ndarray, jnp.floating] # or jnp.float32, jnp.float64, etc
        label: Annotated[jnp.ndarray, jnp.integer] # or jnp.uint8, jnp.uint32, etc

    Fields with both shape and type constraints:

        image: Annotated[jnp.ndarray, (50, 150, 3), jnp.floating]
        label: Annotated[jnp.ndarray, (10,), jnp.integer]

    Where the annotations are order-invariant and both optional.
    """

    def __post_init__(self):
        """Validate after construction.

        We raise assertion errors in only two scenarios:
        - A field has a dtype that's not a subtype of the annotated dtype.
        - A field has a shape that's not consistent with the annotated shape."""

        assert dataclasses.is_dataclass(self)

        hint_from_name = get_type_hints(type(self), include_extras=True)
        batch_axes: Optional[Tuple[int, ...]] = None

        # Batch axes for child/nested elements.
        child_batch_axes_list: List[Tuple[int, ...]] = []

        # For each field...
        for field in dataclasses.fields(self):

            type_hint = hint_from_name[field.name]
            value = self.__getattribute__(field.name)

            if isinstance(value, ArrayAnnotationMixin):
                child_batch_axes = getattr(value, "__batch_axes__")
                if child_batch_axes is not None:
                    child_batch_axes_list.append(child_batch_axes)
                continue

            # Check for metadata from `typing.Annotated` value! Skip if no annotation.
            if not hasattr(type_hint, "__metadata__"):
                continue
            metadata: Tuple[Any, ...] = type_hint.__metadata__
            assert (
                len(metadata) <= 2
            ), "We expect <= 2 metadata items; only shape and dtype are expected."

            # Check data type.
            metadata_dtype = tuple(filter(_is_dtype, metadata))
            if len(metadata_dtype) > 0 and hasattr(value, "dtype"):
                (dtype,) = metadata_dtype
                assert jnp.issubdtype(
                    value.dtype, dtype
                ), f"Mismatched dtype, expected {dtype} but got {value.dtype}."

            # Shape checks.
            metadata_shape = tuple(filter(_is_shape, metadata))
            shape: Optional[Tuple[int, ...]] = None
            if isinstance(value, float):
                shape = ()
            elif hasattr(value, "shape"):
                shape = value.shape
            if len(metadata_shape) > 0 and shape is not None:
                # Get expected shape, sans batch axes.
                (expected_shape,) = metadata_shape

                # Actual shape should be expected shape prefixed by some batch axes.
                if len(expected_shape) > 0:
                    shape_suffix = shape[-len(expected_shape) :]
                    assert (
                        shape_suffix == expected_shape
                    ), f"Trailing shape dimensions did not match: expected {expected_shape} but got {shape_suffix}."
                    field_batch_axes = shape[: -len(expected_shape)]
                else:
                    field_batch_axes = shape

                if batch_axes is None:
                    batch_axes = field_batch_axes
                else:
                    assert (
                        batch_axes == field_batch_axes
                    ), f"Batch axis mismatch: {batch_axes} and {field_batch_axes}."

        # Check child batch axes: any batch axes present in the parent should be present
        # in the children as well.
        for child_batch_axes in child_batch_axes_list:
            assert (
                len(child_batch_axes) >= len(batch_axes)
                and child_batch_axes[: len(batch_axes)] == batch_axes
            ), f"Child batch axes {child_batch_axes} don't match parent axes {batch_axes}."

        object.__setattr__(self, "__batch_axes__", batch_axes)

    def get_batch_axes(self) -> Tuple[int, ...]:
        """Return any leading batch axes (which should be shared across all contained
        arrays)."""

        batch_axes = getattr(self, "__batch_axes__")
        assert batch_axes is not None
        return batch_axes