from llm import LLMASP, LLMHandler
from asp import Solver

def marketplace_example(model, server):
    user_input = "I would like some cooking ideas for a dessert with apples and for a main plate with meat."
    llm = LLMHandler(model, server)
    solver = Solver()
    llmasp = LLMASP("./specifications/application_marketplace.yml", "./specifications/behavior_translator.yml", llm, solver)
    llmasp.run(user_input, verbose=1)