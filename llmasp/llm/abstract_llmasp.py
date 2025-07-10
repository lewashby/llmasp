from abc import ABC, abstractmethod
from typing import Any

class AbstractLLMASP(ABC):
    """
    Abstract base class for LLMASP pipeline.
    """
    def __init__(self, config_file: str, behavior_file: str, llm: Any, solver: Any):
        self.config = self.load_file(config_file)
        self.behavior = self.load_file(behavior_file)
        self.llm = llm
        self.solver = solver

    @abstractmethod
    def load_file(path: str):
        """
        Load a configuration or behavior file.
        """
        pass

    @abstractmethod
    def asp_to_natural(self, *args, **kwargs):
        """
        Convert ASP facts to natural language.
        """
        pass

    @abstractmethod
    def natural_to_asp(self, *args, **kwargs):
        """
        Convert natural language to ASP facts.
        """
        pass

    @abstractmethod
    def run(self, input: Any, verbose: int):
        """
        Run the LLMASP pipeline.
        """
        pass