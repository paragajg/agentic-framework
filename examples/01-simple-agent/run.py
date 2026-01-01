"""
Simple Agent Example

Demonstrates basic agent creation and execution with the Agentic Framework.
"""

import asyncio
import os
from pathlib import Path

# This example assumes the framework is installed
# If running from source, adjust import paths as needed


async def main():
    """Run a simple research agent."""
    print("=" * 60)
    print("Simple Agent Example")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        return

    print("\n📝 Task: Research the latest trends in AI agents")
    print("\n🤖 Agent Configuration:")
    print("   - Role: Research")
    print("   - Capabilities: web_search, summarize")
    print("   - LLM: Anthropic Claude")

    # Example of what the framework would do:
    # 1. Spawn a research agent
    # 2. Bind skills and tools
    # 3. Execute the research task
    # 4. Return structured output

    print("\n🔄 Execution Flow:")
    print("   [1/3] Spawning research agent...")
    print("   [2/3] Executing research task...")
    print("   [3/3] Processing results...")

    # Simulated output (in real usage, this would come from the agent)
    print("\n✅ Results:")
    print("""
    Research Summary:
    - AI agents are increasingly using multi-agent architectures
    - LLM orchestration frameworks are gaining adoption
    - Key trends: tool use, planning, memory systems
    - Enterprise focus: governance, observability, safety

    Sources: 3 web pages analyzed
    Confidence: 0.85
    """)

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - See examples/02-multi-step-workflow/ for orchestration")
    print("  - See examples/03-custom-skill/ for creating skills")
    print("  - Read docs/getting-started.md for full guide")


if __name__ == "__main__":
    asyncio.run(main())
