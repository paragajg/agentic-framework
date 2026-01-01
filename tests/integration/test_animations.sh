#!/bin/bash
# Quick Animation Test Script
# Runs various animation tests to verify everything works

set -e

echo "🎨 Kautilya Animation Test Suite"
echo "================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No .venv found. Please run: uv venv && source .venv/bin/activate"
    exit 1
fi

echo ""
echo "Select a test to run:"
echo ""
echo "  1. Full Animation Demo (all animations)"
echo "  2. Welcome Screen only"
echo "  3. Spinners Collection"
echo "  4. Tool Execution Visualizer"
echo "  5. Success/Error Celebrations"
echo "  6. Quick Smoke Test (fast)"
echo "  7. Run All Tests"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "Running full animation demo..."
        echo "This will show all animations in sequence."
        echo ""
        python -m kautilya.animations_demo
        ;;
    2)
        echo ""
        echo "Testing Welcome Screen..."
        python -c "
from rich.console import Console
from kautilya.animations import WelcomeScreen

console = Console()
WelcomeScreen.show(
    console,
    llm_enabled=True,
    mcp_running=True,
    version='1.0.0',
    animate=True
)
"
        ;;
    3)
        echo ""
        echo "Testing Spinner Collection..."
        python -c "
from rich.console import Console
from kautilya.animations import ModernSpinner
import time

console = Console()

spinner_types = ['dots', 'pulse', 'arrows', 'grow']

for spinner_type in spinner_types:
    spinner = ModernSpinner(
        console,
        f'Testing {spinner_type} spinner',
        spinner_type=spinner_type,
        style='cyan'
    )

    spinner.start()
    time.sleep(2.0)
    spinner.stop(f'{spinner_type.capitalize()} complete')
    time.sleep(0.5)
"
        ;;
    4)
        echo ""
        echo "Testing Tool Execution Visualizer..."
        python -c "
from rich.console import Console
from kautilya.animations import ToolExecutionVisualizer
import time

console = Console()

# Show execution
ToolExecutionVisualizer.show_execution(
    console,
    'web_search',
    args={'query': 'AI frameworks', 'max_results': 10}
)

time.sleep(1.5)

# Show result
ToolExecutionVisualizer.show_result(
    console,
    'web_search',
    success=True,
    duration=1.47,
    summary='Found 10 results from 3 sources'
)
"
        ;;
    5)
        echo ""
        echo "Testing Success/Error Celebrations..."
        python -c "
from rich.console import Console
from kautilya.animations import Celebration
import time

console = Console()

# Success
console.print('\n[dim]Success example:[/dim]')
time.sleep(0.5)

Celebration.success(
    console,
    'Agent created successfully!',
    confetti=True
)

time.sleep(2.0)

# Error
console.print('\n[dim]Error example:[/dim]')
time.sleep(0.5)

Celebration.error(
    console,
    'Failed to connect to OpenAI API',
    details='API key not found. Set OPENAI_API_KEY=sk-...'
)
"
        ;;
    6)
        echo ""
        echo "Running quick smoke test..."
        python -c "
from rich.console import Console
from kautilya.animations import (
    ModernSpinner,
    GradientText,
    Celebration
)
import time

console = Console()

# Test 1: Gradient
print('Test 1: Gradient Text')
text = GradientText.apply('Kautilya - Agentic Framework', gradient='cyberpunk')
console.print(text)
console.print('[green]✓ Gradient OK[/green]\n')
time.sleep(0.5)

# Test 2: Spinner
print('Test 2: Modern Spinner')
spinner = ModernSpinner(console, 'Loading', 'pulse')
spinner.start()
time.sleep(1.0)
spinner.stop('Loaded')
console.print('[green]✓ Spinner OK[/green]\n')
time.sleep(0.5)

# Test 3: Celebration
print('Test 3: Success Celebration')
Celebration.success(console, 'All tests passed!', confetti=False)
console.print('[green]✓ Celebration OK[/green]\n')

console.print('[bold green]All smoke tests passed! ✨[/bold green]')
"
        ;;
    7)
        echo ""
        echo "Running all tests in sequence..."
        echo ""

        # Run each test with a short delay
        for i in {2..5}; do
            echo "Running test $i..."
            $0 <<< "$i"
            echo ""
            sleep 1
        done

        echo "[bold green]All tests complete! ✨[/bold green]"
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-7."
        exit 1
        ;;
esac

echo ""
echo "✅ Test complete!"
echo ""
echo "To run full demo: python -m kautilya.animations_demo"
echo "To see docs: less docs/CLI_ANIMATIONS_GUIDE.md"
echo "To see integration: less ANIMATION_INTEGRATION_EXAMPLE.py"
echo ""
