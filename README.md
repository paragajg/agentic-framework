# Agentic Framework

> Enterprise-grade multi-agent orchestration platform for building LLM-powered workflows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🔄 **LLM-Agnostic**: Support for 6 providers (Anthropic, OpenAI, Azure, Gemini, Ollama, vLLM)
- 📝 **YAML Workflows**: Declarative multi-step agent workflows with typed artifacts
- 🛠️ **Deterministic Skills**: Sandboxed Python functions with JIT loading
- 🔌 **MCP Integration**: Safe external tool access via Model Context Protocol
- 📊 **Full Observability**: Metrics, logging, tracing, and provenance tracking
- 🔐 **Enterprise Ready**: RBAC, audit trails, human-in-the-loop approvals
- 🎯 **Typed Artifacts**: JSON Schema validated outputs between agents

## 🚀 Quick Start

### Installation

**From Git (recommended for latest):**
```bash
pip install git+https://github.com/your-org/agentic-framework.git@v1.0.0
```

**From Source:**
```bash
git clone https://github.com/your-org/agentic-framework.git
cd agentic-framework
pip install -e .
```

### Your First Agent

```bash
# Set your LLM API key
export ANTHROPIC_API_KEY="your-key-here"

# Verify installation
agentctl --version

# Create an agent
agentctl agent new research-agent --role research --capabilities web_search,summarize

# Try an example
cd examples/01-simple-agent
python run.py
```

### Your First Workflow

```yaml
# workflow.yaml
manifest_id: simple-research
name: Simple Research Workflow
version: "1.0.0"

steps:
  - id: research
    role: research
    capabilities: [web_search, summarize]
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  - id: synthesize
    role: synthesis
    inputs:
      - name: research
        source: previous_step
    outputs: [final_report]
    timeout: 20
```

```bash
# Run the workflow
agentctl manifest run workflow.yaml --input "Research AI agent trends"
```

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Architecture Overview](docs/architecture.md)
- [Building Workflows](docs/manifests.md)
- [Creating Skills](docs/skills.md)
- [MCP Integration](docs/mcp.md)
- [API Reference](docs/api-reference.md)

## 🎯 Examples

Check out [examples/](examples/) for complete working examples:

| Example | Description | Difficulty |
|---------|-------------|------------|
| [01-simple-agent](examples/01-simple-agent/) | Single agent with skills | Beginner |
| 02-multi-step-workflow | Sequential agent pipeline | Intermediate |
| 03-custom-skill | Build your own skills | Intermediate |
| 04-mcp-integration | External tool integration | Advanced |

## 🏗️ Architecture

```
User → Lead Agent/Orchestrator → Subagents (Research, Verify, Code, Synthesis)
     → Typed Artifacts → Code Executor → Memory Service → Final Output
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Orchestrator** | Workflow execution engine with YAML manifest support |
| **Subagent Manager** | Isolated agent contexts with capability-based access |
| **Memory Service** | Multi-tier storage (Redis, Postgres, Milvus, S3) |
| **MCP Gateway** | Tool catalog, discovery, and runtime proxy |
| **Code Executor** | Sandboxed skill execution with JIT loading |
| **CLI (agentctl)** | Developer tool for agent/workflow management |

## 🔧 Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for services)
- Redis, PostgreSQL, Milvus (via Docker)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/agentic-framework.git
cd agentic-framework

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start infrastructure services
docker-compose up -d

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black --line-length 100 .

# Type checking
mypy --strict .

# Linting
ruff check .
```

## 🏢 Production Deployment

### Docker Compose (Local/Dev)

```bash
docker-compose up -d
```

### Kubernetes (Production)

```bash
# Using Helm
helm install agentic-framework ./infra/helm/agentic-framework \
  --set llm.provider=anthropic \
  --set llm.apiKeySecret=anthropic-key
```

## 📊 Project Structure

```
agentic-framework/
├── adapters/              # LLM provider adapters
├── orchestrator/          # Workflow orchestration engine
├── subagent-manager/      # Subagent lifecycle management
├── memory-service/        # Multi-tier memory storage
├── mcp-gateway/           # MCP tool gateway
├── code-exec/             # Skill executor & sandbox
├── tools/                 # CLI utilities (agentctl)
├── examples/              # Example projects
├── docs/                  # Documentation
├── tests/                 # Test suite
└── infra/                 # Infrastructure as code
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit (`git commit -m 'feat: add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), and [anyio](https://anyio.readthedocs.io/)
- Supports [Anthropic Claude](https://www.anthropic.com/), [OpenAI GPT](https://openai.com/), and more
- Implements [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/agentic-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/agentic-framework/discussions)

## 🗺️ Roadmap

See [CHANGELOG.md](CHANGELOG.md) for version history and [GitHub Projects](https://github.com/your-org/agentic-framework/projects) for upcoming features.

---

**Made with ❤️ by the Agentic Framework community**
