from __future__ import annotations

import dataclasses

import jax
import pytest
from jax import numpy as jnp

import jax_dataclasses as jdc


def test_jit_0():
    def func(x: int, y: int) -> jax.Array:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return jnp.full(shape=(x,), fill_value=y)

    assert func(3, 4).shape == (3,)


def test_jit_1():
    @jdc.jit
    def func(x: int, y: int) -> jax.Array:
        assert not isinstance(x, int)
        assert not isinstance(y, int)
        return jnp.full(shape=(x,), fill_value=y)

    with pytest.raises(TypeError):
        assert func(3, 4).shape == (3,)


def test_jit_2():
    @jdc.jit
    def func(x: jdc.Static[int], y: int) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        return jnp.full(shape=(x,), fill_value=y)

    assert func(3, 4).shape == (3,)


def test_jit_3():
    @jdc.jit
    def func(x: jdc.Static[int], /, y: int) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        return jnp.full(shape=(x,), fill_value=y)

    assert func(3, 4).shape == (3,)


def test_jit_4():
    @jdc.jit
    def func(*, x: jdc.Static[int], y: int) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        return jnp.full(shape=(x,), fill_value=y)

    assert func(x=3, y=4).shape == (3,)


def test_jit_5():
    @jdc.jit
    def func(x: jdc.Static[int], y: int, /, *, z: jdc.Static[int]) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        assert isinstance(z, int)
        return jnp.full(shape=(x + z,), fill_value=y)

    assert func(2, 4, z=1).shape == (3,)


def test_jit_6():
    @jdc.jit
    def func(x: jdc.Static[int], y: int, *, z: jdc.Static[int]) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        assert isinstance(z, int)
        return jnp.full(shape=(x + z,), fill_value=y)

    assert func(2, 4, z=1).shape == (3,)


def test_jit_7():
    @jdc.jit
    def func(x: jdc.Static[int], y: int, z: jdc.Static[int], /) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        assert isinstance(z, int)
        return jnp.full(shape=(x + z,), fill_value=y)

    assert func(2, 4, 1).shape == (3,)


def test_jit_no_annotation():
    @jdc.jit
    def func(x: jdc.Static[int], y, z: jdc.Static[int], /) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        assert isinstance(z, int)
        return jnp.full(shape=(x + z,), fill_value=y)

    assert func(2, 4, 1).shape == (3,)


def test_jit_donate_buffer():
    @jdc.jit(donate_argnums=(1,))
    def func(x: jdc.Static[int], y: int, z: jdc.Static[int], /) -> jax.Array:
        assert isinstance(x, int)
        assert not isinstance(y, int)
        assert isinstance(z, int)
        out = jnp.full(shape=(x + z,), fill_value=y)

        # Shape matches `y`, so we should be able to reuse the donated buffer.
        return jnp.sum(out)

    assert func(2, 4, 1).shape == ()


def test_jit_forward_ref():
    @jdc.jit
    def func(xz: jdc.Static[SomeConfig], y: int, /) -> jax.Array:
        assert not isinstance(y, int)
        return jnp.full(shape=(xz.x + xz.z,), fill_value=y)

    assert func(SomeConfig(2, 1), 4).shape == (3,)


def test_jit_lambda():
    assert jdc.jit(lambda x, y: x + y)(jnp.zeros(3), jnp.ones(3)).shape == (3,)


@dataclasses.dataclass(frozen=True)
class SomeConfig:
    x: int
    z: int
