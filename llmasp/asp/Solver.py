from clingo import Control
from dumbo_asp.primitives.models import Model

def logger(code, msg):
    return

class Context:
    @staticmethod
    def min(a, b):
        return a if a < b else b

class Solver:

    def solve(self, program, arguments=["--opt-strategy=usc,k,0,5" ,"--opt-usc-shrink=rgs"], timeout=2):

        results: list = []
        handle = None
        def on_model(m):
            results.append(Model.of_atoms(m.symbols(shown=True)))
        
        control = Control(arguments, logger=logger)
        control.add(f"{program}")
        control.ground([("base", [])], context=Context)

        with control.solve(on_model=on_model, async_=True) as handle:
            handle.wait(timeout)
            handle.cancel()
            handle = handle.get()
            
        model = results[0] if len(results) > 0 else Model.empty()
        result = model.as_facts.split('\n') if len(model) > 0 else []
        
        return result, handle.interrupted, handle.satisfiable