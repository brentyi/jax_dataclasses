"""Tests for optional shape and type annotation features."""

import numpy as onp
import pytest
from jax import numpy as jnp
from typing_extensions import Annotated

from jax_dataclasses import ArrayAnnotationMixin, pytree_dataclass


@pytree_dataclass
class MnistStruct(ArrayAnnotationMixin):
    image: Annotated[
        jnp.ndarray,
        (28, 28),
        jnp.floating,
    ]
    label: Annotated[
        jnp.ndarray,
        (10,),
        jnp.integer,
    ]


@pytree_dataclass
class MnistStructPartial(ArrayAnnotationMixin):
    image_shape_only: Annotated[
        jnp.ndarray,
        (28, 28),
    ]
    label_dtype_only: Annotated[
        jnp.ndarray,
        jnp.integer,
    ]


def test_valid():
    data = MnistStruct(
        image=onp.zeros((28, 28), dtype=onp.float32),
        label=onp.zeros((10,), dtype=onp.uint8),
    )
    assert data.get_batch_axes() == ()

    data = MnistStruct(
        image=onp.zeros((5, 28, 28), dtype=onp.float32),
        label=onp.zeros((5, 10), dtype=onp.uint8),
    )
    assert data.get_batch_axes() == (5,)

    data = MnistStruct(
        image=onp.zeros((5, 7, 28, 28), dtype=onp.float32),
        label=onp.zeros((5, 7, 10), dtype=onp.uint8),
    )
    assert data.get_batch_axes() == (5, 7)

    data = MnistStructPartial(
        image_shape_only=onp.zeros((7, 28, 28), dtype=onp.float32),
        label_dtype_only=onp.zeros((70), dtype=onp.int32),
    )
    assert data.get_batch_axes() == (7,)


def test_shape_mismatch():
    with pytest.raises(AssertionError):
        MnistStruct(
            image=onp.zeros((7, 32, 32), dtype=onp.float32),
            label=onp.zeros((7, 10), dtype=onp.uint8),
        )

    with pytest.raises(AssertionError):
        MnistStructPartial(
            image_shape_only=onp.zeros((7, 32, 32), dtype=onp.float32),
            label_dtype_only=onp.zeros((7, 10), dtype=onp.uint8),
        )


def test_batch_axis_mismatch():
    with pytest.raises(AssertionError):
        MnistStruct(
            image=onp.zeros((5, 7, 28, 28), dtype=onp.float32),
            label=onp.zeros((7, 10), dtype=onp.uint8),
        )


def test_dtype_mismatch():
    with pytest.raises(AssertionError):
        MnistStruct(
            image=onp.zeros((7, 28, 28), dtype=onp.uint8),
            label=onp.zeros((7, 10), dtype=onp.uint8),
        )

    with pytest.raises(AssertionError):
        MnistStructPartial(
            image_shape_only=onp.zeros((7, 28, 28), dtype=onp.float32),
            label_dtype_only=onp.zeros((7, 10), dtype=onp.float32),
        )


def test_nested():
    @pytree_dataclass
    class Parent(ArrayAnnotationMixin):
        x: Annotated[jnp.integer, ()]
        child: MnistStruct

    # OK
    assert (
        Parent(
            x=onp.zeros((7,), dtype=onp.float32),
            child=MnistStruct(
                image=onp.zeros((7, 28, 28), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            ),
        ).get_batch_axes()
        == (7,)
    )

    # Batch axis mismatch
    with pytest.raises(AssertionError):
        Parent(
            x=onp.zeros((5,), dtype=onp.float32),
            child=MnistStruct(
                image=onp.zeros((7, 28, 28), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            ),
        )

    # Type error
    with pytest.raises(AssertionError):
        Parent(
            x=onp.zeros((7,), dtype=onp.float32),
            child=MnistStruct(
                image=onp.zeros((7, 28, 28), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.float32),
            ),
        )
