"""Tests for optional shape and type annotation features."""

import jax
import numpy as onp
import pytest
from jax import numpy as jnp
from typing_extensions import Annotated

import jax_dataclasses as jdc


@jdc.pytree_dataclass
class MnistStruct(jdc.EnforcedAnnotationsMixin):
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


@jdc.pytree_dataclass
class MnistStructPartial(jdc.EnforcedAnnotationsMixin):
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
    @jdc.pytree_dataclass
    class Parent(jdc.EnforcedAnnotationsMixin):
        x: Annotated[jnp.integer, ()]
        child: MnistStruct

    # OK
    assert Parent(
        x=onp.zeros((7,), dtype=onp.float32),
        child=MnistStruct(
            image=onp.zeros((7, 28, 28), dtype=onp.float32),
            label=onp.zeros((7, 10), dtype=onp.uint8),
        ),
    ).get_batch_axes() == (7,)

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


def test_scalar():
    @jdc.pytree_dataclass
    class ScalarContainer(jdc.EnforcedAnnotationsMixin):
        scalar: Annotated[jnp.ndarray, ()]  # () => scalar shape

    assert ScalarContainer(scalar=5.0).get_batch_axes() == ()
    assert ScalarContainer(scalar=onp.zeros((5,))).get_batch_axes() == (5,)


def test_grad():
    @jdc.pytree_dataclass
    class Vector3(jdc.EnforcedAnnotationsMixin):
        parameters: Annotated[jnp.ndarray, (3,)]

    # Make sure we can compute gradients wrt annotated dataclasses.
    grad = jax.grad(lambda x: jnp.sum(x.parameters))(Vector3(onp.zeros(3)))
    onp.testing.assert_allclose(grad.parameters, onp.ones((3,)))


# This test currently breaks -- shape assertions on instantiation makes it impossible to
# compute some more complex Jacobians.
#
# Some options for fixing: adding some way to temporarily disable validation, or moving
# away from validation on instantiation to validation only when `.get_batch_axes()` is
# called. Either should be fairly straightforward, but this is fairly niche, produces (in my
# opinion) unintuitive Pytree structures, and is easy to work around, so marking as a
# no-fix for now.
#
# def test_jacobians():
#     @jdc.pytree_dataclass
#     class Vector3(ArrayAnnotationMixin):
#         parameters: Annotated[jnp.ndarray, (3,)]
#
#     @jdc.pytree_dataclass
#     class Vector4(ArrayAnnotationMixin):
#         parameters: Annotated[jnp.ndarray, (4,)]
#
#     def vec4_from_vec3(vec3: Vector3) -> Vector4:
#         return Vector4(onp.zeros((4,)))
#
#     def vec3_from_vec4(vec4: Vector4) -> Vector3:
#         return Vector3(onp.zeros((3,)))
#
#     jac = jax.jacfwd(vec4_from_vec3)(Vector3(onp.zeros((3,))))
#     jac = jax.jacfwd(vec3_from_vec4)(Vector4(onp.zeros((4,))))
