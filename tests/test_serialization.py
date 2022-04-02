"""Tests for serialization using `flax.serialization`.
"""

import flax
import jax
import numpy as onp

import jax_dataclasses as jdc


def _assert_pytree_allclose(x, y):
    jax.tree_multimap(
        lambda *arrays: onp.testing.assert_allclose(arrays[0], arrays[1]), x, y
    )


def test_serialization():
    @jdc.pytree_dataclass
    class A:
        field1: int
        field2: int
        field3: bool = jdc.static_field()

    obj = A(field1=5, field2=3, field3=True)

    _assert_pytree_allclose(
        obj,
        flax.serialization.from_bytes(obj, flax.serialization.to_bytes(obj)),
    )
