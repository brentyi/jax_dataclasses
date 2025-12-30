"""Tests for optional shape and type annotation features."""

import jax
import numpy as onp
import pytest
from jax import numpy as jnp
from jaxtyping import Float, Integer, Shaped
from typing_extensions import Annotated

import jax_dataclasses as jdc


@jdc.pytree_dataclass
class MnistStruct(jdc.EnforcedAnnotationsMixin):
    image: Annotated[
        onp.ndarray,
        (..., 28, 28),
        jnp.floating,
    ]
    label: Annotated[
        onp.ndarray,
        (..., 10),
        jnp.integer,
    ]


@jdc.pytree_dataclass
class JaxTypingMnistStruct(jdc.EnforcedAnnotationsMixin):
    image: Float[onp.ndarray, "28 28"]
    label: Integer[onp.ndarray, "10"]


@jdc.pytree_dataclass
class MnistStructPartial(jdc.EnforcedAnnotationsMixin):
    image_shape_only: Annotated[
        onp.ndarray,
        (28, 28),  # Ellipsis will be appended automatically.
    ]
    label_dtype_only: Annotated[
        onp.ndarray,
        jnp.integer,
    ]


@jdc.pytree_dataclass
class JaxTypingMnistStructPartial(jdc.EnforcedAnnotationsMixin):
    image_shape_only: Shaped[onp.ndarray, "28 28"]
    label_dtype_only: Integer[onp.ndarray, "..."]


def test_valid() -> None:
    for struct, partial_struct in (
        (MnistStruct, MnistStructPartial),
        (JaxTypingMnistStruct, JaxTypingMnistStructPartial),
    ):
        data = struct(
            image=onp.zeros((28, 28), dtype=onp.float32),
            label=onp.zeros((10,), dtype=onp.uint8),
        )
        assert data.get_batch_axes() == ()

        data = struct(
            image=onp.zeros((5, 28, 28), dtype=onp.float32),
            label=onp.zeros((5, 10), dtype=onp.uint8),
        )
        assert data.get_batch_axes() == (5,)

        data = struct(
            image=onp.zeros((5, 7, 28, 28), dtype=onp.float32),
            label=onp.zeros((5, 7, 10), dtype=onp.uint8),
        )
        assert data.get_batch_axes() == (5, 7)

        data_partial = partial_struct(
            image_shape_only=onp.zeros((7, 28, 28), dtype=onp.float32),
            label_dtype_only=onp.zeros((70), dtype=onp.int32),
        )
        assert data_partial.get_batch_axes() == (7,)


def test_shape_mismatch() -> None:
    for struct, partial_struct in (
        (MnistStruct, MnistStructPartial),
        (JaxTypingMnistStruct, JaxTypingMnistStructPartial),
    ):
        with pytest.raises(AssertionError):
            struct(
                image=onp.zeros((7, 32, 32), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            )

        with pytest.raises(AssertionError):
            partial_struct(
                image_shape_only=onp.zeros((7, 32, 32), dtype=onp.float32),
                label_dtype_only=onp.zeros((7, 10), dtype=onp.uint8),
            )


def test_batch_axis_mismatch() -> None:
    for struct in (MnistStruct, JaxTypingMnistStruct):
        with pytest.raises(AssertionError):
            struct(
                image=onp.zeros((5, 7, 28, 28), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            )


def test_dtype_mismatch() -> None:
    for struct, partial_struct in (
        (MnistStruct, MnistStructPartial),
        (JaxTypingMnistStruct, JaxTypingMnistStructPartial),
    ):
        with pytest.raises(AssertionError):
            struct(
                image=onp.zeros((7, 28, 28), dtype=onp.uint8),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            )

        with pytest.raises(AssertionError):
            partial_struct(
                image_shape_only=onp.zeros((7, 28, 28), dtype=onp.float32),
                label_dtype_only=onp.zeros((7, 10), dtype=onp.float32),
            )


def test_nested() -> None:
    for struct in (MnistStruct, JaxTypingMnistStruct):

        @jdc.pytree_dataclass
        class Parent(jdc.EnforcedAnnotationsMixin):
            x: Annotated[onp.ndarray, jnp.floating, ()]
            child: struct

        # OK
        assert Parent(
            x=onp.zeros((7,), dtype=onp.float32),
            child=struct(
                image=onp.zeros((7, 28, 28), dtype=onp.float32),
                label=onp.zeros((7, 10), dtype=onp.uint8),
            ),
        ).get_batch_axes() == (7,)

        # Batch axis mismatch
        with pytest.raises(AssertionError):
            Parent(
                x=onp.zeros((5,), dtype=onp.float32),
                child=struct(
                    image=onp.zeros((7, 28, 28), dtype=onp.float32),
                    label=onp.zeros((7, 10), dtype=onp.uint8),
                ),
            )

        # Type error
        with pytest.raises(AssertionError):
            Parent(
                x=onp.zeros((7,), dtype=onp.float32),
                child=struct(
                    image=onp.zeros((7, 28, 28), dtype=onp.float32),
                    label=onp.zeros((7, 10), dtype=onp.float32),
                ),
            )


def test_scalar() -> None:
    @jdc.pytree_dataclass
    class ScalarContainer(jdc.EnforcedAnnotationsMixin):
        scalar: Annotated[onp.ndarray, ()]  # () => scalar shape

    assert ScalarContainer(scalar=5.0).get_batch_axes() == ()  # type: ignore
    assert ScalarContainer(scalar=onp.zeros((5,))).get_batch_axes() == (5,)

    @jdc.pytree_dataclass
    class JaxTypingScalarContainer(jdc.EnforcedAnnotationsMixin):
        scalar: Float[onp.ndarray, ""]  # () => scalar shape

    assert JaxTypingScalarContainer(scalar=onp.array(5.0)).get_batch_axes() == ()
    assert JaxTypingScalarContainer(scalar=onp.zeros((5,))).get_batch_axes() == (5,)


def test_grad() -> None:
    @jdc.pytree_dataclass
    class Vector3(jdc.EnforcedAnnotationsMixin):
        parameters: Annotated[onp.ndarray, (3,)]

    # Make sure we can compute gradients wrt annotated dataclasses.
    grad = jax.grad(lambda x: jnp.sum(x.parameters))(Vector3(onp.zeros(3)))
    onp.testing.assert_allclose(grad.parameters, onp.ones((3,)))


def test_unannotated() -> None:
    @jdc.pytree_dataclass
    class Test(jdc.EnforcedAnnotationsMixin):
        a: onp.ndarray

    with pytest.raises(AssertionError):
        Test(onp.zeros((2, 1, 2, 3, 5, 7, 9))).get_batch_axes()


def test_middle_batch_axes() -> None:
    @jdc.pytree_dataclass
    class Test(jdc.EnforcedAnnotationsMixin):
        a: Annotated[onp.ndarray, (3, ..., 5, 7, 9)]

    test = Test(onp.zeros((3, 1, 2, 3, 5, 7, 9)))
    assert test.get_batch_axes() == (1, 2, 3)

    with pytest.raises(AssertionError):
        Test(onp.zeros((2, 1, 2, 3, 5, 7, 9)))
    with pytest.raises(AssertionError):
        Test(onp.zeros((3, 1, 2, 3, 5, 7)))

    @jdc.pytree_dataclass
    class JaxTypingTest(jdc.EnforcedAnnotationsMixin):
        a: Float[onp.ndarray, "3 ... 5 7 9"]

    test = JaxTypingTest(onp.zeros((3, 1, 2, 3, 5, 7, 9)))
    assert test.get_batch_axes() == (1, 2, 3)

    with pytest.raises(AssertionError):
        JaxTypingTest(onp.zeros((2, 1, 2, 3, 5, 7, 9)))
    with pytest.raises(AssertionError):
        JaxTypingTest(onp.zeros((3, 1, 2, 3, 5, 7)))


# This test currently breaks -- shape assertions on instantiation makes it impossible to
# compute some more complex Jacobians.
#
# Some options for fixing: adding some way to temporarily disable validation, or moving
# away from validation on instantiation to validation only when `.get_batch_axes()` is
# called. Either should be fairly straightforward, but this is fairly niche, produces (in my
# opinion) unintuitive Pytree structures, and is easy to work around, so marking as a
# no-fix for now.
#
# def test_jacobians() -> None:
#     @jdc.pytree_dataclass
#     class Vector3(ArrayAnnotationMixin):
#         parameters: Annotated[onp.ndarray, (3,)]
#
#     @jdc.pytree_dataclass
#     class Vector4(ArrayAnnotationMixin):
#         parameters: Annotated[onp.ndarray, (4,)]
#
#     def vec4_from_vec3(vec3: Vector3) -> Vector4:
#         return Vector4(onp.zeros((4,)))
#
#     def vec3_from_vec4(vec4: Vector4) -> Vector3:
#         return Vector3(onp.zeros((3,)))
#
#     jac = jax.jacfwd(vec4_from_vec3)(Vector3(onp.zeros((3,))))
#     jac = jax.jacfwd(vec3_from_vec4)(Vector4(onp.zeros((4,))))
