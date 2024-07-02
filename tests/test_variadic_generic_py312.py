# mypy: ignore-errors
#
# PEP 695 generics aren't yet supported in mypy.

from __future__ import annotations

from typing import Generic, TypeVarTuple

import jax_dataclasses as jdc

Ts = TypeVarTuple("Ts")


@jdc.pytree_dataclass
class Args(Generic[*Ts]):
    @staticmethod
    @jdc.jit
    def make(args: jdc.Static[tuple[*Ts]]) -> tuple[Args[*Ts], tuple[*Ts]]:
        return Args(), args


def test0() -> None:
    assert Args.make((1, 2, 3))[1] == (1, 2, 3)


@jdc.pytree_dataclass
class Args2[*T]:
    @staticmethod
    @jdc.jit
    def make[*T_](args: jdc.Static[tuple[*T_]]) -> tuple[Args2[*T_], tuple[*T_]]:
        return Args2(), args


def test1() -> None:
    assert Args2.make((1, 2, 3))[1] == (1, 2, 3)
