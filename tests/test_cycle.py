from __future__ import annotations

from typing import Tuple

from jax import numpy as jnp

import jax_dataclasses as jdc


def test_cycle() -> None:
    @jdc.pytree_dataclass
    class TreeNode:
        content: jnp.ndarray
        children: Tuple[TreeNode, ...]

    a = TreeNode(content=jnp.zeros(3), children=())
    b = TreeNode(content=jnp.ones(3), children=(a,))

    # Not a cycle. OK!
    with jdc.copy_and_mutate(b, validate=False) as b:
        b.children = (a, a)

    # Cycle. Ideally this should raise an error, but the way duplicate nodes in pytrees
    # are possible makes robust cycle detection under linear space/time constraints a
    # hassle. So instead we currently do nothing.
    with jdc.copy_and_mutate(b, validate=False) as b:
        b.children[0].children = (b,)
