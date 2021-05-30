## jax_dataclasses

Library for using dataclasses as JAX PyTrees.

Key features:

- PyTree registration; automatic generation of flatten/unflatten ops.
- Static analysis-friendly. Works out of the box with tools like `mypy` and
  `jedi`.
- Support for serialization via `flax.serialization`.

## Usage

#### Basic

`jax_dataclasses` is meant to be a drop-in replacement for
`dataclasses.dataclass`:

- <code>jax_dataclasses.<strong>dataclass</strong></code> has the same interface
  as `dataclasses.dataclass`, but also register a class as a PyTree.
- <code>jax_dataclasses.<strong>static_field</strong></code> has the same
  interface as `dataclasses.field`, but will also mark the field as static. In a
  PyTree node, static fields are treated as part of the treedef instead of as a
  child of the node.

We also provide several aliases:
`jax_dataclasses.[field, asdict, astuples, is_dataclass, replace]` are all
identical to their counterparts in the standard dataclasses library.

#### Mutations

All dataclasses are automatically marked as frozen and thus immutable. We do,
however, provide an interface that will (a) make a copy of a PyTree and (b)
return a context in which any of that copy's contained dataclasses are
temporarily mutable:

```python
from jax import numpy as jnp
import jax_dataclasses

@jax_dataclasses.dataclass
class Node:
  child: jnp.ndarray

obj = Node(child=jnp.zeros(3))

with jax_dataclasses.copy_and_mutate(obj) as obj_updated:
  # Make mutations to the dataclass.
  # Also does input validation: if the treedef of `obj` and `obj_updated` don't
  # match, an AssertionError will be raised.
  obj_updated.child = jnp.ones(3)

print(obj)
print(obj_updated)
```

## Motivation

For compatibility with function transformations in JAX (jit, grad, vmap, etc),
arguments and return values must all be
[PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) containers.
Dataclasses, by default, are not.

A few great solutions exist for automatically integrating dataclass-style
objects into PyTree structures, notably
[`chex.dataclass`](https://github.com/deepmind/chex) and
[`flax.struct`](https://github.com/google/flax). This library implements another
one.

**Why not use `chex.dataclass`?**

`chex.dataclass` is handy and lightweight, but currently lacks support for:

- Static fields: parameters that are either non-differentiable or simply not
  arrays.
- Serialization using `flax.serialization`. This is really handy when parameters
  needed to be saved to disk!

**Why not use `flax.struct`?**

`flax.struct` addresses the two points above, but both it and `chex.dataclass`:

- Lack support for static analysis and type-checking. Static analysis for
  libraries like `dataclasses` and `attrs` tends to rely on tooling-specific
  custom plugins, which doesn't exist for either `chex.dataclass` or
  `flax.struct`.
- Make modifying deeply nested dataclasses fairly frustrating. Both introduce a
  `.replace(self, ...)` method to dataclasses that's a bit more convenient than
  the traditional `dataclasses.replace(obj, ...)` API, but this becomes really
  cumbersome to use when dataclasses are nested. Fixing this is the goal of
  `jax_dataclasses.copy_and_mutate()`.
