import inspect
from typing import Any, Callable, Optional, Sequence, TypeVar, Union, cast, overload

import jax
from jaxlib import xla_client as xc

from ._dataclasses import JDC_STATIC_MARKER
from ._get_type_hints import get_type_hints_partial

CallableType = TypeVar("CallableType", bound=Callable)


@overload
def jit(
    fun: CallableType,
    *,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> CallableType: ...


@overload
def jit(
    fun: None = None,
    *,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Callable[[CallableType], CallableType]: ...


def jit(
    fun: Optional[CallableType] = None,
    *,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Union[CallableType, Callable[[CallableType], CallableType]]:
    """Light wrapper around `jax.jit`, with usability and type checking improvements.

    Three differences:
        - We remove the `static_argnums` and `static_argnames` parameters. Instead,
          static arguments can be specified in type annotations with
          `jax_dataclasses.Static[]`.
        - Instead of `jax.stages.Wrapped`, the return callable type is annotated to
          match the input callable type. This will improve autocomplete and type
          checking in most situations.
        - Similar to `@dataclasses.dataclass`, return a decorator if `fun` isn't passed
          in. This is convenient for avoiding `@functools.partial()`.
    """

    def wrap(fun: CallableType) -> CallableType:
        signature = inspect.signature(fun)

        # Mark any inputs annotated with jax_dataclasses.Static[] as static.
        static_argnums = []
        static_argnames = []
        hint_from_name = get_type_hints_partial(fun, include_extras=True)
        for i, param in enumerate(signature.parameters.values()):
            name = param.name
            if name not in hint_from_name:
                continue
            hint = hint_from_name[name]
            if hasattr(hint, "__metadata__") and JDC_STATIC_MARKER in hint.__metadata__:
                if param.kind is param.POSITIONAL_ONLY:
                    static_argnums.append(i)
                else:
                    static_argnames.append(name)

        return cast(
            CallableType,
            jax.jit(
                fun,
                static_argnums=static_argnums if len(static_argnums) > 0 else None,
                static_argnames=static_argnames if len(static_argnames) > 0 else None,
                device=device,
                backend=backend,
                donate_argnums=donate_argnums,
                inline=inline,
                keep_unused=keep_unused,
                abstracted_axes=abstracted_axes,
            ),
        )

    if fun is None:
        return wrap
    else:
        return wrap(fun)
