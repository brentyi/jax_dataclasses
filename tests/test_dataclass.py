"""Tests for standard jdc.pytree_dataclass features. Initialization, flattening, unflattening,
static fields, etc.
"""

import jax
import numpy as onp
import pytest
from jax import tree_util

import jax_dataclasses as jdc


def _assert_pytree_allclose(x, y):
    tree_util.tree_map(
        lambda *arrays: onp.testing.assert_allclose(arrays[0], arrays[1]), x, y
    )


def test_init():
    @jdc.pytree_dataclass
    class A:
        field1: int
        field2: int

    assert A(field1=5, field2=3) == A(5, 3)

    with pytest.raises(TypeError):
        # Not enough arguments
        A(field1=5)


def test_default_arg():
    @jdc.pytree_dataclass
    class A:
        field1: int
        field2: int = 3

    assert A(field1=5, field2=3) == A(5, 3) == A(field1=5) == A(5)


def test_flatten():
    @jdc.pytree_dataclass
    class A:
        field1: float
        field2: float

    @jax.jit
    def jitted_sum(obj: A) -> float:
        return obj.field1 + obj.field2

    _assert_pytree_allclose(jitted_sum(A(5.0, 3.0)), 8.0)


def test_unflatten():
    @jdc.pytree_dataclass
    class A:
        field1: float
        field2: float

    @jax.jit
    def construct_A(a: float) -> A:
        return A(field1=a, field2=a * 2.0)

    _assert_pytree_allclose(A(1.0, 2.0), construct_A(1.0))


def test_static_field():
    @jdc.pytree_dataclass
    class A:
        field1: float
        field2: float
        field3: jdc.Static[bool]

    @jax.jit
    def jitted_op(obj: A) -> float:
        if obj.field3:
            return obj.field1 + obj.field2
        else:
            return obj.field1 - obj.field2

    with pytest.raises(ValueError):
        # Cannot map over pytrees with different treedefs
        _assert_pytree_allclose(A(1.0, 2.0, False), A(1.0, 2.0, True))

    _assert_pytree_allclose(jitted_op(A(5.0, 3.0, True)), 8.0)
    _assert_pytree_allclose(jitted_op(A(5.0, 3.0, False)), 2.0)


def test_static_field_deprecated():
    @jdc.pytree_dataclass
    class A:
        field1: float
        field2: float
        field3: bool = jdc.static_field()  # type: ignore

    @jax.jit
    def jitted_op(obj: A) -> float:
        if obj.field3:
            return obj.field1 + obj.field2
        else:
            return obj.field1 - obj.field2

    with pytest.raises(ValueError):
        # Cannot map over pytrees with different treedefs
        _assert_pytree_allclose(A(1.0, 2.0, False), A(1.0, 2.0, True))

    _assert_pytree_allclose(jitted_op(A(5.0, 3.0, True)), 8.0)
    _assert_pytree_allclose(jitted_op(A(5.0, 3.0, False)), 2.0)


def test_no_init():
    @jdc.pytree_dataclass
    class A:
        field1: float
        field2: float = jdc.field()
        field3: bool = jdc.static_field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "field3", False)

    @jax.jit
    def construct_A(a: float) -> A:
        return A(field1=a, field2=a * 2.0)

    assert construct_A(5.0).field3 is False
