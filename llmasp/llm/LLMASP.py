import yaml
import re

from .AbstractLLMASP import AbstractLLMASP
from .LLMHandler import LLMHandler

class LLMASP(AbstractLLMASP):
    
    def __init__(self, config_file: str, behavior_file: str, behavior: str, llm: LLMHandler, solver):
        super().__init__(config_file, behavior_file, llm, solver)
        self.behavior = self.behaviors[behavior]

    def load_file(self, config_file: str):
        return yaml.load(open(config_file, "r"), Loader=yaml.Loader)
    
    def __prompt(self, role: str, content: str):
        return { "role": role, "content": content }

    def __create_queries(self, user_input: str):
        queries = {}
        context = self.behavior["datalog"]["context"]
        for index, query in enumerate(self.config["preprocessing"]):
            queries[index] = []
            queries[index].append(self.__prompt("system", self.behavior["datalog"]["behavior"]))
            format, system_input = list(query.items())[0]
            real_context = re.sub(r"\{format\}", format, context)
            queries[index].append(self.__prompt("system", real_context))
            queries[index].append(self.__prompt("system", system_input))
            queries[index].append(self.__prompt("user", f"USER_INPUT: {user_input}"))
        return queries
    
    def __get_property(self, properties, key, is_fact=False):
        if is_fact:
            property = list(filter(lambda x: next(iter(x)).split("(")[0] == key, properties))[0]        
        else:
            property = list(filter(lambda x: next(iter(x)) == key, properties))[0]
        property_key = next(iter(property))
        property_value = list(property.values())[0]
        return property_key, property_value
    
    def asp_to_natural(self, history: dict, facts:list):

        def group_by_fact(facts: list) -> dict:
            grouped = {}
            for f in facts:
                name = f.split("(")[0]
                grouped.setdefault(name, []).append(f)
            return grouped
        grouped_facts = group_by_fact(facts)

        queries = [x for v in history.values() for x in v]
        context = self.behavior["natural-language"]["context"]
        responses = []
        _, format = self.__get_property(self.config["postprocessing"], "_")
        _, final_response = self.__get_property(self.config["postprocessing"], "summary")
        _, fact_translation = self.__get_property(self.config["postprocessing"], "translation")
        real_context = re.sub(r"\{format\}", format, context)
        for fact_name in grouped_facts:
            f_translation_key, f_translation_value = self.__get_property(self.config["postprocessing"], fact_name, is_fact=True)
            fact_translation = re.sub(r"\{fact\}", fact_translation, f_translation_key)
            fact_translation = re.sub(r"\{meaning\}", fact_translation, f_translation_value)
            res = self.llm.call([*queries, *[
                    self.__prompt("system", real_context),
                    self.__prompt("system", fact_translation),
                    self.__prompt("user", f"[FACTS]{"\n".join(grouped_facts[fact_name])}[/FACTS]"),
                ]])
            responses.append(res)
            print("fact: ", fact_name)
            print(res)
            print("------------------------------------------------")
        final_response = re.sub(r"\{response\}", final_response, "\n".join(responses))
        out = [self.__prompt("user", final_response)]
        return self.llm.call(out)

    def natural_to_asp(self, user_input:str):
        queries = self.__create_queries(user_input)
        asp_input = ""
        for q in queries.values():
            facts = self.llm.call(q)
            facts = re.findall(r"\b[a-zA-Z][\w_]*\([^)]*\)\.", facts)
            facts = "\n".join(facts)
            asp_input = f"{asp_input}\n{facts}"
            q.append(self.__prompt("assistant", facts))
        print(facts)
        knowledge_base = self.config["knowledge_base"]
        asp_input = f"{asp_input}\n{knowledge_base}"

        return asp_input, queries

    def run(self):
        user_input = input("input: ")
        asp_input, history = self.natural_to_asp(user_input)

        result, _, _ = self.solver.solve(asp_input)
        print(result)

        response = self.asp_to_natural({}, result)
        print(response)
