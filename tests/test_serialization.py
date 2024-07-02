"""Tests for serialization using `flax.serialization`."""

import flax
import numpy as onp
from jax import tree_util

import jax_dataclasses as jdc


def _assert_pytree_allclose(x, y) -> None:
    tree_util.tree_map(
        lambda *arrays: onp.testing.assert_allclose(arrays[0], arrays[1]), x, y
    )


def test_serialization() -> None:
    @jdc.pytree_dataclass
    class A:
        field1: int
        field2: int
        field3: jdc.Static[bool]

    obj = A(field1=5, field2=3, field3=True)

    _assert_pytree_allclose(
        obj,
        flax.serialization.from_bytes(obj, flax.serialization.to_bytes(obj)),
    )
