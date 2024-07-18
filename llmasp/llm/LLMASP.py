import yaml
import re

from .AbstractLLMASP import AbstractLLMASP
from .LLMHandler import LLMHandler

class LLMASP(AbstractLLMASP):
    
    def __init__(self, config_file: str, behavior_file: str, llm: LLMHandler, solver):
        super().__init__(config_file, behavior_file, llm, solver)
        db_file = self.load_file(self.config["database"])
        self.database = db_file["database"]

    def __get_atom_name(self, atom: str):
        return atom.split("(")[0]

    def __prompt(self, role: str, content: str):
        return { "role": role, "content": content }

    def __create_queries(self, user_input: str):
        queries = []
        context = self.behavior["preprocessing"]["context"]
        mapping = self.behavior["preprocessing"]["mapping"]
        mapping = re.sub(r"\{input\}", user_input, mapping)
        _, application_context = self.__get_property(self.config["preprocessing"], "_")
        real_context = re.sub(r"\{context\}", application_context, context)
        for query in self.config["preprocessing"]:
            key, value = list(query.items())[0]
            if (key != "_"):
                application_mapping = re.sub(r"\{instructions\}", value, mapping)
                application_mapping = re.sub(r"\{atom\}", key, application_mapping)
                queries.append([
                    self.__prompt("system", self.behavior["preprocessing"]["init"]),
                    self.__prompt("system", real_context),
                    self.__prompt("system", application_mapping)
                ])
        return queries
    
    def __get_property(self, properties, key, is_fact=False):
        if is_fact:
            property = list(filter(lambda x: self.__get_atom_name(next(iter(x))) == key, properties))[0]        
        else:
            property = list(filter(lambda x: next(iter(x)) == key, properties))[0]
        property_key = next(iter(property))
        property_value = list(property.values())[0]
        return property_key, property_value
    
    def load_file(self, config_file: str):
        return yaml.load(open(config_file, "r"), Loader=yaml.Loader)
    
    def asp_to_natural(self, history: list, facts:list):

        def group_by_fact(facts: list) -> dict:
            grouped = {}
            for f in facts:
                name = self.__get_atom_name(f)
                grouped.setdefault(name, []).append(f)
            return grouped
        grouped_facts = group_by_fact(facts)

        responses = []
        queries = [x for v in history for x in v]
        context = self.behavior["postprocessing"]["context"]
        
        _, application_context = self.__get_property(self.config["postprocessing"], "_")
        application_context = re.sub(r"\{context\}", application_context, context)
        final_response = self.behavior["postprocessing"]["summarize"]
        for fact_name in grouped_facts:
            fact_translation = self.behavior["postprocessing"]["mapping"]
            f_translation_key, f_translation_value = self.__get_property(self.config["postprocessing"], fact_name, is_fact=True)
            fact_translation = re.sub(r"\{atom\}", f_translation_key, fact_translation)
            fact_translation = re.sub(r"\{intructions\}", f_translation_value, fact_translation)
            fact_translation = re.sub(r"\{facts\}", "\n".join(grouped_facts[fact_name]), fact_translation)
            res = self.llm.call([*queries, *[
                    self.__prompt("system", self.behavior["postprocessing"]["init"]),
                    self.__prompt("system", application_context),
                    self.__prompt("system", fact_translation),
                ]])
            responses.append(res)
        
        final_response = re.sub(r"\{responses\}", "\n".join(responses), final_response)
        return self.llm.call([self.__prompt("system", application_context), self.__prompt("system", final_response)])

    def natural_to_asp(self, user_input:str):
        queries = self.__create_queries(user_input)
        created_facts = ""
        for q in queries:
            facts = self.llm.call(q)
            facts = re.findall(r"\b[a-zA-Z][\w_]*\([^)]*\)\.", facts)
            facts = "\n".join(facts)
            created_facts = f"{created_facts}\n{facts}"
            q.append(self.__prompt("assistant", facts))
        asp_input = f"{created_facts}\n{self.database}\n{self.config["knowledge_base"]}"

        return created_facts, asp_input, queries

    def run(self, user_input, verbose=0):
        logs = []
        output = ""
        logs.append(f"input: {user_input}")
        created_facts, asp_input, history = self.natural_to_asp(user_input)
        logs.append(f"extracted facts: {created_facts}")

        result, _, _ = self.solver.solve(asp_input)
        if (len(result) == 0):
            logs.extend(["answer set: not found", "outp: not found"])
        else:
            logs.append(f"answer set: {result}")
            response = self.asp_to_natural(history, result)
            logs.append(f"output: {response}")
            output = response
        if (verbose == 1):
            print("\n\n".join(logs))
        return output
