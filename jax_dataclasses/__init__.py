from dataclasses import asdict, astuple, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING

from ._copy_and_mutate import copy_and_mutate

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
    # For the future, dataclass transforms may also be worth looking into:
    # https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
    from dataclasses import dataclass as pytree_dataclass
    from dataclasses import field as static_field
else:
    from ._dataclasses import pytree_dataclass, static_field

from ._enforced_annotations import EnforcedAnnotationsMixin

__all__ = [
    "asdict",
    "astuple",
    "field",
    "fields",
    "is_dataclass",
    "replace",
    "copy_and_mutate",
    "pytree_dataclass",
    "static_field",
    "EnforcedAnnotationsMixin",
]
