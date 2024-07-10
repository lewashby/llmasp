
from abc import ABC, abstractmethod

class AbstractLLMASP(ABC):

    def __init__(self, config_file:str, behavior_file:str, llm, solver):
        self.config = self.load_file(config_file)
        self.behaviors = self.load_file(behavior_file)
        self.llm = llm
        self.solver = solver

    @abstractmethod
    def load_file(path:str):
        pass

    @abstractmethod
    def asp_to_natural(self):
        pass
    
    @abstractmethod
    def natural_to_asp(self):
        pass
    
    @abstractmethod
    def run(self):
        pass