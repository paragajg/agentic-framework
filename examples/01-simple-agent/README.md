# Simple Agent Example

This example demonstrates how to create a basic research agent using the Agentic Framework.

## Overview

Creates a single agent that can:
- Use LLM for research tasks
- Bind skills (text summarization)
- Access MCP tools (web search)

## Prerequisites

```bash
# Ensure framework is installed
pip install -e ../..

# Set LLM API key
export ANTHROPIC_API_KEY="your-key-here"
```

## Running the Example

```bash
# Run the agent
python run.py
```

## What This Example Shows

1. **Agent Configuration**: How to define an agent with role and capabilities
2. **Skill Binding**: Attaching a summarization skill to the agent
3. **MCP Tool Integration**: Using external tools like web search
4. **Simple Execution**: Running a basic research task

## Files

- `run.py` - Main script to run the agent
- `agent_config.yaml` - Agent configuration
- `README.md` - This file

## Expected Output

The agent will:
1. Accept a research query
2. Use web search to find information
3. Summarize the findings
4. Return structured output

## Next Steps

- See `02-multi-step-workflow/` for orchestrating multiple agents
- See `03-custom-skill/` for creating your own skills
