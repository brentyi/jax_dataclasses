## jax_dataclasses

![build](https://github.com/brentyi/jax_dataclasses/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jax_dataclasses/workflows/mypy/badge.svg?branch=main)
![lint](https://github.com/brentyi/jax_dataclasses/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jax_dataclasses/branch/main/graph/badge.svg?token=fFSx7CeKlW)](https://codecov.io/gh/brentyi/jax_dataclasses)

`jax_dataclasses` provides a wrapper around `dataclasses.dataclass` for use in
JAX, which enables automatic support for:

- [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html) registration. This
  allows dataclasses to be used at API boundaries in JAX. (necessary for
  function transformations, JIT, etc)
- Serialization via `flax.serialization`.
- Static analysis with tools like `mypy`, `jedi`, `pyright`, etc. (including for
  constructors)
- (Optional) Shape and data-type annotations, runtime checks.

Notably, `jax_dataclasses` is designed to work seamlessly with tooling that
relies on static analysis. (`mypy`, `jedi`, etc)

Heavily influenced by some great existing work (the obvious one being
`flax.struct.dataclass`); see [Alternatives](#alternatives) for comparisons.

### Installation

```bash
pip install jax_dataclasses
```

### Core interface

`jax_dataclasses` is meant to provide a drop-in replacement for
`dataclasses.dataclass`:

- <code>jax_dataclasses.<strong>pytree_dataclass</strong></code> has the same
  interface as `dataclasses.dataclass`, but also registers the target class as a
  pytree container.
- <code>jax_dataclasses.<strong>static_field</strong></code> has the same
  interface as `dataclasses.field`, but will also mark the field as static. In a
  pytree node, static fields will be treated as part of the treedef instead of
  as a child of the node; all fields that are not explicitly marked static
  should contain arrays or child nodes.

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

@jax_dataclasses.pytree_dataclass
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

### Shape and type annotations

As an optional feature, we introduce
<code>jax_dataclasses.<strong>ArrayAnnotationMixin</strong></code> to enable
automatic shape and data-type validation.

We can start by importing the `Annotated` type:

```python
# Python >=3.9
from typing import Annotated

# Backport
from typing_extensions import Annotated
```

We can then add shape annotations:

```python
@jax_dataclasses.pytree_dataclass
class MnistStruct(jax_dataclasses.ArrayAnnotationMixin):
    image: Annotated[
        jnp.ndarray,
        (28, 28),
    ]
    label: Annotated[
        jnp.ndarray,
        (10,),
    ]
```

Or data-type annotations:

```python
    image: Annotated[
        jnp.ndarray,
        jnp.float32,
    ]
    label: Annotated[
        jnp.ndarray,
        jnp.integer,
    ]
```

Or both (note that annotations are order-invariant):

```python
    image: Annotated[
        jnp.ndarray,
        (28, 28),
        jnp.float32,
    ]
    label: Annotated[
        jnp.ndarray,
        (10,),
        jnp.integer,
    ]
```

Then, assuming we've constrained both the shape and data-type, we get array
validation on instantiation and access to a `.get_batch_axes()` method for
grabbing any common prefixes in contained array shapes:

```python
# OK
struct = MnistStruct(
  image=onp.zeros((28, 28), dtype=onp.float32),
  label=onp.zeros((10,), dtype=onp.uint8),
)
print(struct.get_batch_axes()) # Prints ()

# OK
struct = MnistStruct(
  image=onp.zeros((32, 28, 28), dtype=onp.float32),
  label=onp.zeros((32, 10), dtype=onp.uint8),
)
print(struct.get_batch_axes()) # Prints (32,)

# AssertionError on instantiation because of type mismatch
MnistStruct(
  image=onp.zeros((28, 28), dtype=onp.float32),
  label=onp.zeros((10,), dtype=onp.float32),
)

# AssertionError on instantiation because of shape mismatch
MnistStruct(
  image=onp.zeros((28, 28), dtype=onp.float32),
  label=onp.zeros((5,), dtype=onp.uint8),
)

# AssertionError on instantiation because of batch axis mismatch
struct = MnistStruct(
  image=onp.zeros((64, 28, 28), dtype=onp.float32),
  label=onp.zeros((32, 10), dtype=onp.uint8),
)
```

### Alternatives

A few other solutions exist for automatically integrating dataclass-style
objects into pytree structures. Great ones include:
[`chex.dataclass`](https://github.com/deepmind/chex),
[`flax.struct`](https://github.com/google/flax), and
[`tjax.dataclass`](https://github.com/NeilGirdhar/tjax). These all influenced
this library.

The main differentiators of `jax_dataclasses` are:

- **Static analysis support.** `tjax` has a custom mypy plugin to enable type
  checking, but isn't supported by other tools. `flax.struct` implements the
  [`dataclass_transform`](https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md)
  spec proposed by pyright, but isn't supported by other tools. Because
  `@jax_dataclasses.pytree_dataclass` has the same API as
  `@dataclasses.dataclass`, it can include pytree registration behavior at
  runtime while being treated as the standard decorator during static analysis.
  This means that all static checkers, language servers, and autocomplete
  engines that support the standard `dataclasses` library should work out of the
  box with `jax_dataclasses`.

- **Nested dataclasses.** Making replacements/modifications in deeply nested
  dataclasses can be really frustrating. The three alternatives all introduce a
  `.replace(self, ...)` method to dataclasses that's a bit more convenient than
  the traditional `dataclasses.replace(obj, ...)` API for shallow changes, but
  still becomes really cumbersome to use when dataclasses are nested.
  `jax_dataclasses.copy_and_mutate()` is introduced to address this.

- **Static field support.** Parameters that should not be traced in JAX should
  be marked as static. This is supported in `flax`, `tjax`, and
  `jax_dataclasses`, but not `chex`.

- **Serialization.** When working with `flax`, being able to serialize
  dataclasses is really handy. This is supported in `flax.struct` (naturally)
  and `jax_dataclasses`, but not `chex` or `tjax`.

- **Shape and type annotations.** See above.

### Misc

This code was originally written for and factored out of
[jaxfg](http://github.com/brentyi/jaxfg), where
[Nick Heppert](https://github.com/SuperN1ck) provided valuable feedback!
