"""
Solver module for integrating with clingo and dumbo-asp.
"""

from clingo import Control
from dumbo_asp.primitives.models import Model
from typing import List, Tuple, Any


def logger(code, msg):
    """Stub logger for clingo control."""
    return


class Context:
    """Context for ASP solving, providing custom functions."""

    @staticmethod
    def min(a, b):
        return a if a < b else b


class Solver:
    """
    Solver interface for running ASP programs with clingo and dumbo-asp.
    """

    def solve(
        self,
        program: str,
        arguments: List[str] = ["--opt-strategy=usc,k,0,5", "--opt-usc-shrink=rgs"],
        timeout: int = 2,
        context: Any = Context,
    ) -> Tuple[List[str], Any, Any]:
        results: List[Model] = []
        handle = None

        def on_model(m):
            results.append(Model.of_atoms(m.symbols(shown=True)))

        control = Control(arguments, logger=logger)
        control.add(f"{program}")
        control.ground([("base", [])], context=context)

        with control.solve(on_model=on_model, async_=True) as handle:
            handle.wait(timeout)
            handle.cancel()
            handle = handle.get()

        model = results[0] if len(results) > 0 else Model.empty()
        result = model.as_facts.split("\n") if len(model) > 0 else []

        return result, handle.interrupted, handle.satisfiable