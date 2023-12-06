## jax_dataclasses

![build](https://github.com/brentyi/jax_dataclasses/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jax_dataclasses/workflows/mypy/badge.svg?branch=main)
![lint](https://github.com/brentyi/jax_dataclasses/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jax_dataclasses/branch/main/graph/badge.svg?token=fFSx7CeKlW)](https://codecov.io/gh/brentyi/jax_dataclasses)

<!-- vim-markdown-toc GFM -->

- [Overview](#overview)
- [Installation](#installation)
- [Core interface](#core-interface)
- [Static fields](#static-fields)
- [Mutations](#mutations)
- [Alternatives](#alternatives)
- [Misc](#misc)

<!-- vim-markdown-toc -->

### Overview

`jax_dataclasses` provides a simple wrapper around `dataclasses.dataclass` for use in
JAX, which enables automatic support for:

- [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html) registration. This
  allows dataclasses to be used at API boundaries in JAX.
- Serialization via `flax.serialization`.

Distinguishing features include:

- An annotation-based interface for marking static fields.
- Improved ergonomics for "model surgery" in nested structures.

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
`jdc.[field, asdict, astuples, is_dataclass, replace]` are identical to
their counterparts in the standard dataclasses library.

### Static fields

To mark a field as static (in this context: constant at compile-time), we can
wrap its type with <code>jdc.<strong>Static[]</strong></code>:

```python
@jdc.pytree_dataclass
class A:
    a: jax.Array
    b: jdc.Static[bool]
```

In a pytree node, static fields will be treated as part of the treedef instead
of as a child of the node; all fields that are not explicitly marked static
should contain arrays or child nodes.

Bonus: if you like `jdc.Static[]`, we also introduce
<code>jdc.<strong>jit()</strong></code>. This enables use in function
signatures, for example:

```python
@jdc.jit
def f(a: jax.Array, b: jdc.Static[bool]) -> jax.Array:
  ...
```

### Mutations

All dataclasses are automatically marked as frozen and thus immutable (even when
no `frozen=` parameter is passed in). To make changes to nested structures
easier, <code>jdc.<strong>copy_and_mutate</strong></code> (a) makes a copy of a
pytree and (b) returns a context in which any of that copy's contained
dataclasses are temporarily mutable:

```python
import jax
from jax import numpy as jnp
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class Node:
  child: jax.Array

obj = Node(child=jnp.zeros(3))

with jdc.copy_and_mutate(obj) as obj_updated:
  # Make mutations to the dataclass. This is primarily useful for nested
  # dataclasses.
  #
  # Does input validation by default: if the treedef, leaf shapes, or dtypes
  # of `obj` and `obj_updated` don't match, an AssertionError will be raised.
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

You can also eschew the dataclass-style interface entirely;
[see how brax registers pytrees](https://github.com/google/brax/blob/730e05d4af58eada5b49a44e849107d76e386b9a/brax/pytree.py).
This is a reasonable thing to prefer: it requires some floating strings and
breaks things that I care about but you may not (like immutability and
`__post_init__`), but gives more flexibility with custom `__init__` methods.

### Misc

`jax_dataclasses` was originally written for and factored out of
[jaxfg](http://github.com/brentyi/jaxfg), where
[Nick Heppert](https://github.com/SuperN1ck) provided valuable feedback.
