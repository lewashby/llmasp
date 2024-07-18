# LLMASP

LLMASP is a framework that tries to unify the potential of Large Language Models (LLMs) and the reasoning power of Answer Set Programming (ASP), a form of declarative programming oriented towards difficult search problems.

## Installation

Install poetry following the official [documentation](https://python-poetry.org/docs/).

```bash
git clone https://github.com/LewAshby/llmasp.git
cd llmasp
poetry install
```

Download and install Ollama follwing the official [documentation](https://ollama.com/download), then run

```bash
ollama serve
```

## Usage

If you installed Ollama in an external server, first create an SSH local port forwarding.

```bash
ssh -L <local_port>:<remote_server>:<remote_port> <user>@<remote_server>
```

Ollama binds 127.0. 0.1 port 11434 by default

Use the command-line help for usage intructions.

```bash
cd llmasp
poetry run python main.py --help
```

## Acknowledgments

* [**Ollama**](https://ollama.com/)
* [**OpenAI**](https://platform.openai.com/docs/overview)
* [**dumbo-asp**](https://pypi.org/project/dumbo-asp/)

## License

[Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)