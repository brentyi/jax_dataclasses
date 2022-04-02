import jax
import pytest
from jax import numpy as jnp
from typing_extensions import Annotated

import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Node(jdc.EnforcedAnnotationsMixin):
    a: Annotated[jnp.ndarray, (5,), jnp.floating]


def test_vmap():
    with pytest.raises(AssertionError):
        jax.jit(jax.vmap(lambda *unused: None))(Node(jnp.zeros((5,))))

    jax.jit(jax.vmap(lambda *unused: None))(Node(jnp.zeros((5, 5))))
    jax.jit(jax.vmap(lambda *unused: None, in_axes=(None, None, 0, 0)))(
        Node(jnp.zeros(5)),
        Node(jnp.zeros(5)),
        jnp.zeros((5, 100)),
        jnp.zeros((5, 100)),
    )
