from __future__ import annotations

import collections
import sys
from typing import Any, Dict, Type


class _UnresolvableForwardReference:
    def __class_getitem__(cls, item) -> Type[_UnresolvableForwardReference]:
        """__getitem__ passthrough, for supporting generics."""
        return _UnresolvableForwardReference


def get_type_hints_partial(obj, include_extras=False) -> Dict[str, Any]:
    """Adapted from typing.get_type_hints(), but aimed at suppressing errors from not
    (yet) resolvable forward references.

    For example:

        @jdc.pytree_dataclass
        class A:
            x: B
            y: jdc.Static[bool]

        @jdc.pytree_dataclass
        class B:
            x: jnp.ndarray

    Note that the type annotations of `A` need to be parsed by the `pytree_dataclass`
    decorator in order to register the static field, but `B` is not yet defined when the
    decorator is run. We don't actually care about the details of the `B` annotation, so
    we replace it in our annotation dictionary with a dummy value.

    Differences:
        1. `include_extras` must be True.
        2. Only supports types.
        3. Doesn't throw an error when a name is not found. Instead, replaces the value
           with `_UnresolvableForwardReference`.
    """
    assert include_extras
    assert isinstance(obj, type)

    hints = {}
    for base in reversed(obj.__mro__):
        # Replace any unresolvable names with _UnresolvableForwardReference.
        base_globals: Dict[str, Any] = collections.defaultdict(
            lambda: _UnresolvableForwardReference
        )
        base_globals.update(sys.modules[base.__module__].__dict__)

        ann = base.__dict__.get("__annotations__", {})
        for name, value in ann.items():
            if value is None:
                value = type(None)
            if isinstance(value, str):
                value = eval(value, base_globals)
            hints[name] = value
    return hints
