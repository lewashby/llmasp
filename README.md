# LLMASP

LLMASP is a framework that tries to unify the potential of Large Language Models (LLMs) and the reasoning power of Answer Set Programming (ASP), a form of declarative programming oriented towards difficult search problems.

## Installation

Install poetry following the official [documentation](https://python-poetry.org/docs/).

```bash
git clone https://github.com/LewAshby/llmasp.git
cd llmasp
poetry install
```

Download and install Ollama following the official [documentation](https://ollama.com/download), then run

```bash
ollama serve
```

Make sure you downloaded the model before calling it. E.g.

```bash
ollama pull llama3
```

Go to [Ollama library](https://ollama.com/library) for all available models.

## Usage

If you are running Ollama locally, the default IP address + port will be <http://localhost:11434>, for more details about Ollama and OpenAI compability go [here](https://ollama.com/blog/openai-compatibility).
If you installed Ollama in an external server, then create an SSH local port forwarding.

```bash
ssh -L <local_port>:<remote_server>:<remote_port> <user>@<remote_server>
```

Ollama binds 127.0. 0.1 port 11434 by default

Use the command-line help for usage instructions.

```bash
cd llmasp
poetry run python main.py --help
```

Run example case

```bash
poetry run python main.py --example -m llama3 -s http://localhost:11434/v1
```

## Installation from PyPI
To install and use the package directly from PyPI, you can use pip, the Python package installer. Follow the steps from the [package page](https://pypi.org/project/llmasp/).

## Acknowledgments

* [**Ollama**](https://ollama.com/)
* [**OpenAI**](https://platform.openai.com/docs/overview)
* [**dumbo-asp**](https://pypi.org/project/dumbo-asp/)

## License

[Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)