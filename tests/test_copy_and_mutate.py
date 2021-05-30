import dataclasses
from typing import List

import numpy as onp
import pytest
from jax import numpy as jnp

import jax_dataclasses


@jax_dataclasses.dataclass(
    # frozen=True will do nothing for registered dataclasses
    frozen=True
)
class Foo:
    array: jnp.ndarray


@jax_dataclasses.dataclass
class Bar:
    children: List[Foo]
    array: jnp.ndarray
    array_unchanged: jnp.ndarray


def test_copy_and_mutate():
    obj = Bar(
        children=[Foo(array=onp.zeros(3))],
        array=onp.ones(3),
        array_unchanged=onp.ones(3),
    )

    # Registered dataclasses are generally immutable!
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)

    # But we can use a context that copies a dataclass and temporarily makes the copy
    # mutable:
    with jax_dataclasses.copy_and_mutate(obj) as obj:
        # Updates can then very easily be applied!
        obj.array = onp.zeros(3)
        obj.children[0].array = onp.ones(3)

        # Shapes can be validated...
        with pytest.raises(AssertionError):
            obj.children[0].array = onp.ones(1)

        # As well as dtypes
        with pytest.raises(AssertionError):
            obj.children[0].array = onp.ones(3, dtype=onp.int32)

    # Outside of the replace context, the copied object becomes immutable again:
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.children[0].array = onp.ones(3)

    onp.testing.assert_allclose(obj.array, onp.zeros(3))
    onp.testing.assert_allclose(obj.array_unchanged, onp.ones(3))
    onp.testing.assert_allclose(obj.children[0].array, onp.ones(3))
