## jax_dataclasses

![build](https://github.com/brentyi/jax_dataclasses/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jax_dataclasses/workflows/mypy/badge.svg?branch=main)
![lint](https://github.com/brentyi/jax_dataclasses/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jax_dataclasses/branch/main/graph/badge.svg?token=fFSx7CeKlW)](https://codecov.io/gh/brentyi/jax_dataclasses)

<!-- vim-markdown-toc GFM -->

* [Overview](#overview)
* [Installation](#installation)
* [Core interface](#core-interface)
* [Static fields](#static-fields)
* [Mutations](#mutations)
* [Shape and data-type annotations](#shape-and-data-type-annotations)
* [Alternatives](#alternatives)
* [Misc](#misc)

<!-- vim-markdown-toc -->

### Overview

`jax_dataclasses` provides a wrapper around `dataclasses.dataclass` for use in
JAX, which enables automatic support for:

- [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html) registration. This
  allows dataclasses to be used at API boundaries in JAX. (necessary for
  function transformations, JIT, etc)
- Serialization via `flax.serialization`.
- Static analysis with tools like `mypy`, `jedi`, `pyright`, etc. (including for
  constructors)
- Optional shape and data-type annotations, which are checked at runtime.

Heavily influenced by some great existing work (the obvious one being
`flax.struct.dataclass`); see [Alternatives](#alternatives) for comparisons.

### Installation

In Python >=3.7:

```bash
pip install jax_dataclasses
```

We can then import:

```python
import jax_dataclasses as jdc
```

### Core interface

`jax_dataclasses` is meant to provide a drop-in replacement for
`dataclasses.dataclass`: <code>jdc.<strong>pytree_dataclass</strong></code> has
the same interface as `dataclasses.dataclass`, but also registers the target
class as a pytree node.

We also provide several aliases:
`jdc.[field, asdict, astuples, is_dataclass, replace]` are all identical to
their counterparts in the standard dataclasses library.

### Static fields

To mark a field as static (in this context: constant at compile-time), we can
wrap its type with <code>jdc.<strong>Static[]</strong></code>:

```python
@jdc.pytree_dataclass
class A:
    a: jnp.ndarray
    b: jdc.Static[bool]
```

In a pytree node, static fields will be treated as part of the treedef instead
of as a child of the node; all fields that are not explicitly marked static
should contain arrays or child nodes.

### Mutations

All dataclasses are automatically marked as frozen and thus immutable (even when
no `frozen=` parameter is passed in). To make changes to nested structures
easier, <code>jdc.<strong>copy_and_mutate</strong></code> (a) makes a copy of a
pytree and (b) returns a context in which any of that copy's contained
dataclasses are temporarily mutable:

```python
from jax import numpy as jnp
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class Node:
  child: jnp.ndarray

obj = Node(child=jnp.zeros(3))

with jdc.copy_and_mutate(obj) as obj_updated:
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

### Shape and data-type annotations

Subclassing from <code>jdc.<strong>EnforcedAnnotationsMixin</strong></code>
enables automatic shape and data-type validation. Arrays contained within
dataclasses are validated on instantiation and a **`.get_batch_axes()`** method
is exposed for grabbing any common batch axes to the shapes of contained arrays.

We can start by importing the standard `Annotated` type:

```python
# Python >=3.9
from typing import Annotated

# Backport
from typing_extensions import Annotated
```

We can then add shape annotations:

```python
@jdc.pytree_dataclass
class MnistStruct(jdc.EnforcedAnnotationsMixin):
    image: Annotated[
        jnp.ndarray,
        # Note that we can move the expected location of the batch axes by
        # shifting the ellipsis around.
        #
        # If the ellipsis is excluded, we assume batch axes at the start of the
        # shape.
        (..., 28, 28),
    ]
    label: Annotated[
        jnp.ndarray,
        (..., 10),
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
        (..., 28, 28),
        jnp.float32,
    ]
    label: Annotated[
        jnp.ndarray,
        (..., 10),
        jnp.integer,
    ]
```

Then, assuming we've constrained both the shape and data-type:

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
  label=onp.zeros((10,), dtype=onp.float32), # Not an integer type!
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
  `@jdc.pytree_dataclass` has the same API as `@dataclasses.dataclass`, it can
  include pytree registration behavior at runtime while being treated as the
  standard decorator during static analysis. This means that all static
  checkers, language servers, and autocomplete engines that support the standard
  `dataclasses` library should work out of the box with `jax_dataclasses`.

- **Nested dataclasses.** Making replacements/modifications in deeply nested
  dataclasses can be really frustrating. The three alternatives all introduce a
  `.replace(self, ...)` method to dataclasses that's a bit more convenient than
  the traditional `dataclasses.replace(obj, ...)` API for shallow changes, but
  still becomes really cumbersome to use when dataclasses are nested.
  `jdc.copy_and_mutate()` is introduced to address this.

- **Static field support.** Parameters that should not be traced in JAX should
  be marked as static. This is supported in `flax`, `tjax`, and
  `jax_dataclasses`, but not `chex`.

- **Serialization.** When working with `flax`, being able to serialize
  dataclasses is really handy. This is supported in `flax.struct` (naturally)
  and `jax_dataclasses`, but not `chex` or `tjax`.

- **Shape and type annotations.** See above.

You can also eschew the dataclass-style interface entirely;
[see how brax registers pytrees](https://github.com/google/brax/blob/730e05d4af58eada5b49a44e849107d76e386b9a/brax/pytree.py).
This is a reasonable thing to prefer: it requires some floating strings and
breaks things that I care about but you may not (like immutability and
`__post_init__`), but gives more flexibility with custom `__init__` methods.

### Misc

This code was originally written for and factored out of
[jaxfg](http://github.com/brentyi/jaxfg), where
[Nick Heppert](https://github.com/SuperN1ck) provided valuable feedback.
