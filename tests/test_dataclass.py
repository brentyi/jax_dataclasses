"""Tests for standard jax_dataclasses.dataclass features. Initialization, flattening, unflattening,
static fields, etc.
"""

import jax
import numpy as onp
import pytest

import jax_dataclasses


def _assert_pytree_allclose(x, y):
    jax.tree_multimap(
        lambda *arrays: onp.testing.assert_allclose(arrays[0], arrays[1]), x, y
    )


def test_init():
    @jax_dataclasses.dataclass
    class A:
        field1: int
        field2: int

    assert A(field1=5, field2=3) == A(5, 3)

    with pytest.raises(TypeError):
        # Not enough arguments
        A(field1=5)


def test_default_arg():
    @jax_dataclasses.dataclass
    class A:
        field1: int
        field2: int = 3

    assert A(field1=5, field2=3) == A(5, 3) == A(field1=5) == A(5)


def test_flatten():
    @jax_dataclasses.dataclass
    class A:
        field1: float
        field2: float

    @jax.jit
    def jitted_sum(obj: A) -> float:
        return obj.field1 + obj.field2

    _assert_pytree_allclose(jitted_sum(A(5.0, 3.0)), 8.0)


def test_unflatten():
    @jax_dataclasses.dataclass
    class A:
        field1: float
        field2: float

    @jax.jit
    def construct_A(a: float) -> A:
        return A(field1=a, field2=a * 2.0)

    _assert_pytree_allclose(A(1.0, 2.0), construct_A(1.0))


def test_static_field():
    @jax_dataclasses.dataclass
    class A:
        field1: float
        field2: float = jax_dataclasses.field()
        field3: bool = jax_dataclasses.static_field()

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
