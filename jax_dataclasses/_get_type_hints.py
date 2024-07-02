from __future__ import annotations

import collections
import functools
import sys
import types
from types import MethodDescriptorType, MethodWrapperType, WrapperDescriptorType
from typing import Any, Dict, Type


class _UnresolvableForwardReference:
    def __class_getitem__(cls, item) -> Type[_UnresolvableForwardReference]:
        """__getitem__ passthrough, for supporting generics."""
        return _UnresolvableForwardReference


_allowed_types = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    WrapperDescriptorType,
    MethodWrapperType,
    MethodDescriptorType,
)


@functools.lru_cache(maxsize=128)
def get_type_hints_partial(obj, include_extras=False) -> Dict[str, Any]:
    """Adapted from typing.get_type_hints(), but aimed at suppressing errors from not
    (yet) resolvable forward references.

    This function should only be used to search for fields that are annotated with
    `jdc.Static[]`.

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

    # Replace any unresolvable names with _UnresolvableForwardReference.
    base_globals: Dict[str, Any] = collections.defaultdict(
        lambda: _UnresolvableForwardReference
    )
    base_globals.update(__builtins__)  # type: ignore

    # Classes require a special treatment.
    if isinstance(obj, type):
        hints = {}
        for base in reversed(obj.__mro__):
            ann = base.__dict__.get("__annotations__", {})
            if len(ann) == 0:
                continue

            base_globals.update(sys.modules[base.__module__].__dict__)

            for name, value in ann.items():
                if value is None:
                    value = type(None)
                if isinstance(value, str):
                    # The * replace is a hack for variadic generic support.
                    value = value.replace("*", "")
                    value = eval(value, base_globals)
                hints[name] = value
        return hints

    nsobj = obj

    # Find globalns for the unwrapped object.
    while hasattr(nsobj, "__wrapped__"):
        nsobj = nsobj.__wrapped__
    base_globals.update(getattr(nsobj, "__globals__", {}))

    hints = getattr(obj, "__annotations__", None)  # type: ignore
    if hints is None:
        # Return empty annotations for something that _could_ have them.
        if isinstance(obj, _allowed_types):
            return {}
        else:
            raise TypeError(
                "{!r} is not a module, class, method, or function.".format(obj)
            )
    hints = dict(hints)
    for name, value in hints.items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            # The * replace is a hack for variadic generic support.
            value = value.replace("*", "")
            value = eval(value, base_globals)
        hints[name] = value
    return hints
