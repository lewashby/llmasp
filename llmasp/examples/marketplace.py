"""
Example for running the marketplace scenario with LLMASP.
"""

from llmasp.llm import LLMASP, LLMHandler
from llmasp.asp import Solver


def marketplace_example(model: str, server: str):
    """
    Run the marketplace example with a sample user input.
    """
    user_input = "I would like some cooking ideas for a dessert with apples and for a main plate with meat."
    llm = LLMHandler(model, server)
    solver = Solver()
    llmasp = LLMASP("./specifications/application_marketplace.yml", "./specifications/behavior_translator_v2.yml", llm, solver)
    llmasp.run(user_input, verbose=1)