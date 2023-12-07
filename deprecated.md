# Deprecated features

`jax_dataclasses` includes utilities for `__post_init__`-based runtime shape
and datatype annotation. This works as-designed, but we no longer recommend
using it. [jaxtyping](https://github.com/google/jaxtyping) is a reasonable
alternative solution.

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
