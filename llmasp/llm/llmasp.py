"""
LLMASP: Main pipeline for converting natural language to ASP and back using LLMs and an ASP solver.
"""
import yaml
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from yaml.error import YAMLError

from .abstract_llmasp import AbstractLLMASP
from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class LLMASPError(Exception):
    """Base exception for LLMASP errors."""
    pass

class ConfigError(LLMASPError):
    """Configuration related errors."""
    pass

@dataclass
class PreProcessingConfig:
    """Schema for preprocessing configuration."""
    context: str
    mapping: str
    init: str

@dataclass
class PostProcessingConfig:
    """Schema for postprocessing configuration."""
    context: str
    mapping: str
    init: str
    summarize: str

class LLMASP(AbstractLLMASP):
    """
    Main pipeline for LLMASP framework.
    
    Handles the conversion between natural language and ASP, using LLMs for translation
    and an ASP solver for logical inference.
    
    Attributes:
        config: Dictionary containing application configuration
        behavior: Dictionary containing behavior rules
        llm: LLM handler instance
        solver: ASP solver instance
        database: Optional database content from config
    """
    def __init__(self, config_file: str, behavior_file: str, llm: LLMHandler, solver: Any):
        """
        Initialize LLMASP with configuration files and handlers.

        Args:
            config_file: Path to application configuration YAML
            behavior_file: Path to behavior rules YAML
            llm: LLM handler instance
            solver: ASP solver instance
        
        Raises:
            ConfigError: If configuration files are invalid or missing required fields
        """
        try:
            super().__init__(config_file, behavior_file, llm, solver)
            self._validate_config()
            try:
                db_file = self.load_file(self.config["database"])
                self.database = db_file["database"]
            except (KeyError, FileNotFoundError) as e:
                logger.warning(f"Database configuration not found: {e}")
                self.database = ""
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                self.database = ""
        except Exception as e:
            raise ConfigError(f"Failed to initialize LLMASP: {e}")

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        required_config = {"preprocessing", "knowledge_base", "postprocessing"}
        required_behavior = {
            "preprocessing": {"context", "mapping", "init"},
            "postprocessing": {"context", "mapping", "init", "summarize"}
        }
        
        if not all(key in self.config for key in required_config):
            raise ConfigError(f"Missing required config keys: {required_config - set(self.config.keys())}")
            
        for section, fields in required_behavior.items():
            if section not in self.behavior:
                raise ConfigError(f"Missing behavior section: {section}")
            if not all(field in self.behavior[section] for field in fields):
                raise ConfigError(f"Missing required fields in {section}: {fields - set(self.behavior[section].keys())}")

    def __get_atom_name(self, atom: str) -> str:
        """Extract predicate name from atom."""
        return atom.split("(")[0]

    def __prompt(self, role: str, content: str) -> Dict[str, str]:
        """Create a prompt dictionary for LLM."""
        return {"role": role, "content": content}

    def __create_queries(self, user_input: str, single_pass: bool = False) -> List[List[Dict[str, str]]]:
        """
        Create LLM queries for natural language to ASP conversion.
        
        Args:
            user_input: Natural language input
            single_pass: Whether to process all mappings in one query
            
        Returns:
            List of query messages for the LLM
        """
        try:
            queries = []
            context = self.behavior["preprocessing"]["context"]
            mapping = self.behavior["preprocessing"]["mapping"]
            mapping = re.sub(r"\{input\}", user_input, mapping)
            _, application_context = self.__get_property(self.config["preprocessing"], "_")
            real_context = re.sub(r"\{context\}", application_context, context)

            if single_pass:
                formats, instructions = zip(*[
                    (key, value) 
                    for query in self.config["preprocessing"]
                    for key, value in query.items() 
                    if key != "_"
                ])
                application_mapping = re.sub(r"\{instructions\}", "".join(instructions), mapping)
                application_mapping = re.sub(r"\{atom\}", " ".join(formats), application_mapping)
                queries.append(self._create_query_messages(real_context, application_mapping))
            else:
                for query in self.config["preprocessing"]:
                    key, value = list(query.items())[0]
                    if key != "_":
                        application_mapping = re.sub(r"\{instructions\}", value, mapping)
                        application_mapping = re.sub(r"\{atom\}", key, application_mapping)
                        queries.append(self._create_query_messages(real_context, application_mapping))
            
            return queries
        except Exception as e:
            logger.error(f"Error creating queries: {e}")
            raise LLMASPError(f"Failed to create queries: {e}")

    def _create_query_messages(self, context: str, mapping: str) -> List[Dict[str, str]]:
        """Create a list of messages for a single query."""
        return [
            self.__prompt("system", self.behavior["preprocessing"]["init"]),
            self.__prompt("system", context),
            self.__prompt("user", mapping)
        ]

    def __get_property(self, properties: List[Dict[str, Any]], key: str, is_fact: bool = False) -> Tuple[str, Any]:
        """
        Get property from configuration.
        
        Args:
            properties: List of property dictionaries
            key: Property key to find
            is_fact: Whether to match fact names
            
        Returns:
            Tuple of property key and value
        
        Raises:
            LLMASPError: If property not found
        """
        try:
            if is_fact:
                property = next(
                    p for p in properties 
                    if self.__get_atom_name(next(iter(p))) == key
                )
            else:
                property = next(p for p in properties if next(iter(p)) == key)
            
            property_key = next(iter(property))
            property_value = list(property.values())[0]
            return property_key, property_value
        except StopIteration:
            raise LLMASPError(f"Property not found: {key}")
        except Exception as e:
            raise LLMASPError(f"Error getting property: {e}")

    def load_file(self, config_file: str) -> Any:
        """
        Load and validate a YAML configuration file.
        
        Args:
            config_file: Path to YAML file
            
        Returns:
            Parsed YAML content
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {config_file}")
        except YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading {config_file}: {e}")

    def asp_to_natural(
        self, 
        facts: List[str], 
        history: List[Any], 
        use_history: bool = True
    ) -> Tuple[str, Any]:
        """
        Convert ASP facts to natural language using the LLM.
        
        Args:
            facts: List of ASP facts
            history: Previous conversation history
            use_history: Whether to use conversation history
            
        Returns:
            Tuple of (response text, metadata)
            
        Raises:
            LLMASPError: If conversion fails
        """
        try:
            def group_by_fact(facts: List[str]) -> Dict[str, List[str]]:
                """Group facts by predicate name."""
                grouped = {}
                for f in facts:
                    name = self.__get_atom_name(f)
                    grouped.setdefault(name, []).append(f)
                return grouped

            grouped_facts = group_by_fact(facts)
            responses = []
            queries = [x for v in history for x in v] if use_history else []
            
            # Get context and prepare final response template
            context = self.behavior["postprocessing"]["context"]
            _, application_context = self.__get_property(self.config["postprocessing"], "_")
            application_context = re.sub(r"\{context\}", application_context, context)
            final_response = self.behavior["postprocessing"]["summarize"]
            
            # Process each fact group
            for fact_name, fact_group in grouped_facts.items():
                response = self._process_fact_group(
                    fact_name, 
                    fact_group, 
                    queries, 
                    application_context
                )
                responses.append(response)
            
            # Generate final response
            final_response = re.sub(r"\{responses\}", "\n".join(responses), final_response)
            return self.llm.call([
                self.__prompt("system", application_context),
                self.__prompt("user", final_response)
            ])
            
        except Exception as e:
            logger.error(f"Error converting ASP to natural language: {e}")
            raise LLMASPError(f"Failed to convert ASP to natural language: {e}")

    def _process_fact_group(
        self, 
        fact_name: str, 
        facts: List[str], 
        queries: List[Any], 
        context: str
    ) -> str:
        """Process a group of related facts."""
        fact_translation = self.behavior["postprocessing"]["mapping"]
        f_key, f_value = self.__get_property(
            self.config["postprocessing"], 
            fact_name, 
            is_fact=True
        )
        
        # Apply translations
        fact_translation = re.sub(r"\{atom\}", f_key, fact_translation)
        fact_translation = re.sub(r"\{intructions\}", f_value, fact_translation)
        fact_translation = re.sub(r"\{facts\}", "\n".join(facts), fact_translation)
        
        # Get response from LLM
        response, _ = self.llm.call([
            *queries,
            self.__prompt("system", self.behavior["postprocessing"]["init"]),
            self.__prompt("system", context),
            self.__prompt("user", fact_translation),
        ])
        return response

    def natural_to_asp(
        self, 
        user_input: str, 
        single_pass: bool = False, 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, str, List[Any], Any]:
        """
        Convert natural language to ASP facts using the LLM.
        
        Args:
            user_input: Natural language input
            single_pass: Whether to process all mappings in one query
            max_tokens: Optional maximum tokens for LLM response
            
        Returns:
            Tuple of (created facts, ASP input, query history, metadata)
            
        Raises:
            LLMASPError: If conversion fails
        """
        try:
            queries = self.__create_queries(user_input, single_pass=single_pass)
            created_facts = []
            meta = None
            
            for query in queries:
                facts, meta = self.llm.call(query, max_tokens=max_tokens)
                # Extract predicate expressions
                facts = re.findall(r"\b[a-zA-Z][\w_]*\([^)]*\)", facts)
                facts = [f"{f}." for f in facts]
                created_facts.extend(facts)
                query.append(self.__prompt("assistant", "\n".join(facts)))
            
            created_facts_str = "\n".join(created_facts)
            asp_input = f"{created_facts_str}\n{self.database}\n{self.config['knowledge_base']}"
            return created_facts_str, asp_input, queries, meta
            
        except Exception as e:
            logger.error(f"Error converting natural language to ASP: {e}")
            raise LLMASPError(f"Failed to convert natural language to ASP: {e}")

    def run(
        self, 
        user_input: str, 
        single_pass: bool = False, 
        use_history: bool = False, 
        verbose: int = 0
    ) -> Optional[str]:
        """
        Run the complete LLMASP pipeline.
        
        Args:
            user_input: Natural language input
            single_pass: Whether to process all mappings in one query
            use_history: Whether to use conversation history
            verbose: Verbosity level (0 or 1)
            
        Returns:
            Natural language response or None if error occurs
            
        Raises:
            LLMASPError: If pipeline execution fails
        """
        try:
            logs = []
            logs.append(f"input: {user_input}")
            
            # Convert to ASP
            created_facts, asp_input, history, _ = self.natural_to_asp(
                user_input, 
                single_pass=single_pass
            )
            logs.append(f"extracted facts: {created_facts}")
            print(f"extracted facts: {created_facts}")
            
            # Solve ASP program
            result, interrupted, satisfiable = self.solver.solve(asp_input)
            if not result:
                logs.extend(["answer set: not found", "out: not found"])
                return None if not verbose else ""
            
            # Convert back to natural language
            logs.append(f"answer set: {result}")
            response, _ = self.asp_to_natural(result, history, use_history=use_history)
            logs.append(f"output: {response}")
            
            if verbose == 1:
                print("\n\n".join(logs))
                
            return response
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            if verbose == 1:
                print(f"Error: {e}")
            return None
