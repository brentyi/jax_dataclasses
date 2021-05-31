## jax_dataclasses

![build](https://github.com/brentyi/jax_dataclasses/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jax_dataclasses/workflows/mypy/badge.svg?branch=main)
![lint](https://github.com/brentyi/jax_dataclasses/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jax_dataclasses/branch/main/graph/badge.svg?token=fFSx7CeKlW)](https://codecov.io/gh/brentyi/jax_dataclasses)

This library provides a thin wrapper around `dataclasses.dataclass`, which
automatically enables:

- [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html) registration.
  This allows dataclasses to be used at API boundaries in JAX. (necessary for
  function transformations, etc)
- Support for serialization via `flax.serialization`.
- Static analysis-friendly. Works out of the box with tools like `mypy` and
  `jedi`.

This library was heavily influenced by some great existing work; see
[Alternatives](#alternatives) for a comparison.

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

All dataclasses are automatically marked as frozen and thus immutable (even when
no `frozen=` parameter is passed in). To make changes to nested structures
easier, we provide an interface that will (a) make a copy of a pytree and (b)
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
  # Make mutations to the dataclass. This is primarily useful for nested
  # dataclasses.
  #
  # Also does input validation: if the treedef, leaf shapes, or dtypes of `obj`
  # and `obj_updated` don't match, an AssertionError will be raised.
  # This can be disabled with a `validate=False` argument.
  obj_updated.child = jnp.ones(3)

print(obj)
print(obj_updated)
```

### Alternatives

A few other solutions exist for automatically integrating dataclass-style
objects into pytree structures. Great ones include:
[`chex.dataclass`](https://github.com/deepmind/chex),
[`flax.struct`](https://github.com/google/flax), and
[`tjax.dataclass`](https://github.com/NeilGirdhar/tjax). These all influenced
this library.

The main differentiators of `jax_dataclasses` are:

- **Static analysis support.** Libraries like `dataclasses` and `attrs` rely on
  tooling-specific custom plugins for static analysis, which don't exist for
  `chex` or `flax`. `tjax` has a custom mypy plugin to enable type checking, but
  isn't supported by other tools.

  - Because `@jax_dataclasses.dataclass` decorator has the same API as
    `@dataclasses.dataclass`, it can include pytree registration behavior at
    runtime while being treated as the standard decorator during static
    analysis. This means that all static checkers, language servers, and
    autocomplete engines that support the standard `dataclasses` library should
    work out of the box with `jax_dataclasses`.

- **Nested dataclasses.** Making replacements/modifications in deeply nested
  dataclasses is generally very frustrating. The three alternatives all
  introduce a `.replace(self, ...)` method to dataclasses that's a bit more
  convenient than the traditional `dataclasses.replace(obj, ...)` API for
  shallow changes, but still becomes really cumbersome to use when dataclasses
  are nested. `jax_dataclasses.copy_and_mutate()` is introduced to address this.

- **Static field support.** Parameters that should not be traced in JAX should
  be marked as static. This is supported in `flax`, `tjax`, and
  `jax_dataclasses`, but not `chex`.

- **Serialization.** When working with `flax`, being able to serialize
  dataclasses is really handy. This is supported in `flax.struct` (naturally)
  and `jax_dataclasses`, but not `chex` or `tjax`.

### Misc

This code was originally written for and factored out of
[jaxfg](http://github.com/brentyi/jaxfg).
[Nick Heppert](https://github.com/SuperN1ck) provided valuable feedback.
