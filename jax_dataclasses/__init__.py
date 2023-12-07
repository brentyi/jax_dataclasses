from dataclasses import asdict, astuple, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING

from ._copy_and_mutate import copy_and_mutate as copy_and_mutate

if TYPE_CHECKING:
    # Treat our JAX field and dataclass functions as their counterparts from the
    # standard dataclasses library during static analysis
    #
    # Tools like via mypy, jedi, etc generally rely on a lot of special, hardcoded
    # behavior for the standard dataclasses library; this lets us take advantage of all
    # of it.
    #
    # Note that mypy will not follow aliases, so `from dataclasses import dataclass` is
    # preferred over `dataclass = dataclasses.dataclass`.
    #
    # Dataclass transforms serve a similar purpose, but are currently only supported in
    # pyright and pylance.
    # https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
    # `static_field()` is deprecated, but not a lot of code to support, so leaving it
    # for now...
    from dataclasses import dataclass as pytree_dataclass
else:
    from ._dataclasses import pytree_dataclass  # noqa
    from ._dataclasses import deprecated_static_field as static_field  # noqa

from ._dataclasses import Static
from ._enforced_annotations import EnforcedAnnotationsMixin
from ._jit import jit

__all__ = [
    "asdict",
    "astuple",
    "field",
    "fields",
    "is_dataclass",
    "replace",
    "copy_and_mutate",
    "pytree_dataclass",
    "Static",
    "EnforcedAnnotationsMixin",
    "jit",
]
