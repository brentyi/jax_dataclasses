from typing import TYPE_CHECKING

from ._copy_and_mutate import copy_and_mutate

if TYPE_CHECKING:
    # Treat our JAX field and dataclass functions as standard dataclasses for the sake
    # of static analysis.
    #
    # Tools like via mypy, jedi, etc generally rely on a lot of special, hardcoded
    # behavior for the standard dataclasses library; this lets us take advantage of most
    # of it.
    #
    # Note that mypy in particular will not follow aliases, so `from dataclasses import
    # dataclass` is preferred over `dataclass = dataclasses.dataclass`.
    #
    # For the future, this is also an option:
    # https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
    from dataclasses import dataclass
    from dataclasses import field as static_field
else:
    from ._dataclass import dataclass, static_field

from dataclasses import asdict, astuple, field, is_dataclass, replace

__all__ = [
    "copy_and_mutate",
    "dataclass",
    "static_field",
    "asdict",
    "astuples",
    "field",
    "is_dataclass",
    "replace",
]
