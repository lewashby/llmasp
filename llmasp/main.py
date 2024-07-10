from llm import LLMASP, LLMHandler
from asp import Solver

def main():
    llm = LLMHandler("llama3:70b", "http://localhost:11434/v1")
    solver = Solver()
    llmasp = LLMASP("./specifications/marketplace.yml", "./specifications/behaviors.yml", "datalog-translator", llm, solver)
    llmasp.run()

if __name__ == "__main__":
    main()