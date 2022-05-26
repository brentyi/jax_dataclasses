import dataclasses
from typing import Dict, List

import numpy as onp
import pytest
from jax import numpy as jnp

import jax_dataclasses as jdc


def test_copy_and_mutate():
    # frozen=True should do nothing
    @jdc.pytree_dataclass(frozen=True)
    class Foo:
        array: jnp.ndarray

    @jdc.pytree_dataclass
    class Bar:
        children: List[Foo]
        array: jnp.ndarray
        array_unchanged: jnp.ndarray

    obj = Bar(
        children=[Foo(array=onp.zeros(3))],
        array=onp.ones(3),
        array_unchanged=onp.ones(3),
    )

    # Registered dataclasses are initially immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)

    # But we can use a context that copies a dataclass and temporarily makes the copy
    # mutable:
    with jdc.copy_and_mutate(obj) as obj:
        # Updates can then very easily be applied!
        obj.array = onp.zeros(3)
        obj.children[0].array = onp.ones(3)

        # Shapes can be validated...
        with pytest.raises(AssertionError):
            obj.children[0].array = onp.ones(1)

        # As well as dtypes
        with pytest.raises(AssertionError):
            obj.children[0].array = onp.ones(3, dtype=onp.int32)

    # Validation can also be disabled
    with jdc.copy_and_mutate(obj, validate=False) as obj:
        obj.children[0].array = onp.ones(1)
        obj.children[0].array = onp.ones(3)

    # Outside of the replace context, the copied object becomes immutable again:
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.children[0].array = onp.ones(3)

    onp.testing.assert_allclose(obj.array, onp.zeros(3))
    onp.testing.assert_allclose(obj.array_unchanged, onp.ones(3))
    onp.testing.assert_allclose(obj.children[0].array, onp.ones(3))


def test_copy_and_mutate_static():
    @dataclasses.dataclass
    class Inner:
        a: int
        b: int

    @jdc.pytree_dataclass
    class Foo:
        arrays: Dict[str, jnp.ndarray]
        child: Inner = jdc.static_field()

    obj = Foo(arrays={"x": onp.ones(3)}, child=Inner(1, 2))

    # Registered dataclasses are initially immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.child = Inner(5, 6)

    assert obj.child == Inner(1, 2)

    # But can be copied and mutated in a special context
    with jdc.copy_and_mutate(obj) as obj_updated:
        obj_updated.child = Inner(5, 6)

    assert obj.child == Inner(1, 2)
    assert obj_updated.child == Inner(5, 6)
