import pytest
from unittest.mock import MagicMock, patch
from llmasp.llm.llmasp import LLMASP, LLMASPError
from llmasp.asp.solver import Solver
from llmasp.llm.llm_handler import LLMHandler

# --- LLMASP tests ---
def test_llmasp_init_sets_attributes(tmp_path):
    # Create minimal valid config and behavior files
    config_file = tmp_path / "config.yml"
    behavior_file = tmp_path / "behavior.yml"
    config_content = """
preprocessing:
  context: ''
  mapping: ''
  init: ''
knowledge_base: ''
postprocessing:
  context: ''
  mapping: ''
  init: ''
  summarize: ''
"""
    behavior_content = """
preprocessing:
  context: ''
  mapping: ''
  init: ''
postprocessing:
  context: ''
  mapping: ''
  init: ''
  summarize: ''
"""
    config_file.write_text(config_content)
    behavior_file.write_text(behavior_content)
    llm = MagicMock(spec=LLMHandler)
    solver = MagicMock(spec=Solver)
    llmasp = LLMASP(str(config_file), str(behavior_file), llm, solver)
    assert hasattr(llmasp, "config")
    assert hasattr(llmasp, "behavior")
    assert llmasp.llm == llm
    assert llmasp.solver == solver

def test_llmasp_config_error(tmp_path):
    # Pass a non-existent config file
    with pytest.raises(LLMASPError):
        LLMASP("nonexistent.yml", "nonexistent2.yml", MagicMock(), MagicMock())

# --- Solver tests ---
def test_solver_solve_simple():
    solver = Solver()
    # A simple ASP program with one fact
    program = "a."
    result, interrupted, satisfiable = solver.solve(program)
    assert isinstance(result, list)
    assert not interrupted
    assert satisfiable is not None

def test_solver_solve_empty():
    solver = Solver()
    program = ""
    result, interrupted, satisfiable = solver.solve(program)
    assert result == []

# --- LLMHandler tests ---
def test_llmhandler_init():
    with patch("llmasp.llm.llm_handler.OpenAI") as mock_openai:
        handler = LLMHandler(model_name="test-model", server_url="http://test.com", api_key="key")
        assert handler.model == "test-model"
        mock_openai.assert_called()

def test_llmhandler_call():
    with patch("llmasp.llm.llm_handler.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        handler = LLMHandler()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="result"))]
        mock_response.usage = {"tokens": 10}
        mock_client.chat.completions.create.return_value = mock_response
        completion, meta = handler.call([{"role": "user", "content": "hi"}])
        assert completion == "result"
        assert meta == {"tokens": 10}
