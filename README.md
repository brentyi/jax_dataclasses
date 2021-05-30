## jax_dataclasses

![build](https://github.com/brentyi/jax_dataclasses/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jax_dataclasses/workflows/mypy/badge.svg?branch=main)
![lint](https://github.com/brentyi/jax_dataclasses/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jax_dataclasses/branch/main/graph/badge.svg?token=fFSx7CeKlW)](https://codecov.io/gh/brentyi/jax_dataclasses)

For compatibility with function transformations in JAX (jit, grad, vmap, etc),
arguments and return values must all be registered as
[pytree](https://jax.readthedocs.io/en/latest/pytrees.html) containers.
Dataclasses, by default, are not.

This library provides a thin wrapper around `dataclasses.dataclass`, which
automatically enables:

- Pytree registration. This allows dataclasses to be used at API boundaries in
  JAX. (necessary for function transformations, etc)
- Support for serialization via `flax.serialization`.
- Static analysis-friendly. Works out of the box with tools like `mypy` and
  `jedi`.

A few other great solutions exist for automatically integrating dataclass-style
objects into pytree structures, notably
[`chex.dataclass`](https://github.com/deepmind/chex),
[`flax.struct`](https://github.com/google/flax), and
[`tjax.dataclass`](https://github.com/NeilGirdhar/tjax). See
[Alternatives](#alternatives) for notes on differences.

### Installation

```bash
pip install jax_dataclasses
```

### Core interface

`jax_dataclasses` is meant to provide a drop-in replacement for
`dataclasses.dataclass`:

- <code>jax_dataclasses.<strong>dataclass</strong></code> has the same interface
  as `dataclasses.dataclass`, but also registers the target class as a pytree.
- <code>jax_dataclasses.<strong>static_field</strong></code> has the same
  interface as `dataclasses.field`, but will also mark the field as static. In a
  pytree node, static fields are treated as part of the treedef instead of as a
  child of the node.

We also provide several aliases:
`jax_dataclasses.[field, asdict, astuples, is_dataclass, replace]` are all
identical to their counterparts in the standard dataclasses library.

### Mutations

All dataclasses are automatically marked as frozen and thus immutable
(regardless of whether a `frozen=` parameter is passed in). To make changes to
nested structures easier, we provide an interface that will (a) make a copy of a
pytree and (b) return a context in which any of that copy's contained
dataclasses are temporarily mutable:

```python
from jax import numpy as jnp
import jax_dataclasses

@jax_dataclasses.dataclass
class Node:
  child: jnp.ndarray

obj = Node(child=jnp.zeros(3))

with jax_dataclasses.copy_and_mutate(obj) as obj_updated:
  # Make mutations to the dataclass.
  #
  # Also does input validation: if the treedef, leaf shapes, or dtypes of `obj`
  # and `obj_updated` don't # match, an AssertionError will be raised.
  # This can be disabled with a `validate=False` argument.
  obj_updated.child = jnp.ones(3)

print(obj)
print(obj_updated)
```

### Alternatives

`chex.dataclass` is handy and lightweight, but currently lacks support for:

- Static fields: parameters that are either non-differentiable or simply not
  arrays.
- Serialization using `flax.serialization`. This is really handy when parameters
  needed to be saved to disk!

`flax.struct` addresses the two points above, but both it and `chex.dataclass`:

- Lack support for static analysis and type-checking. Static analysis for
  libraries like `dataclasses` and `attrs` tends to rely on tooling-specific
  custom plugins, which don't exist for either `chex.dataclass` or
  `flax.struct`. See `jax_dataclasses/_dataclasses.py` for how we fix this.

`tjax.dataclass` includes a custom `mypy` plugin for type checking, but:

- Doesn't support other static analysis tools. (autocomplete, language servers,
  etc)
- Lacks support for `flax.serialization`.

Finally, all 3 of the above make modifying deeply nested dataclasses really
frustrating. They introduce a `.replace(self, ...)` method to dataclasses that's
a bit more convenient than the traditional `dataclasses.replace(obj, ...)` API
for shallow changes, but still becomes really cumbersome to use when dataclasses
are nested. `jax_dataclasses.copy_and_mutate()` is introduced to address this.
