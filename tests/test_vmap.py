import jax
from jax import numpy as jnp

import jax_dataclasses


@jax_dataclasses.pytree_dataclass
class Node:
    a: jnp.ndarray


def test_vmap():
    jax.jit(jax.vmap(lambda *unused: None))(Node(jnp.zeros(5)))
    jax.jit(jax.vmap(lambda *unused: None, in_axes=(None, None, 0, 0)))(
        Node(jnp.zeros(5)),
        Node(jnp.zeros(5)),
        jnp.zeros((5, 100)),
        jnp.zeros((5, 100)),
    )
