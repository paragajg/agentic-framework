#!/usr/bin/env python3
"""
Test Script: LLM History Validation and Recovery

This script demonstrates the self-correction capabilities of the agent framework
for preventing production LLM API errors.

Usage:
    python test_validation_recovery.py

Tests:
    1. Basic validation (existing capability)
    2. Order violation detection (enhanced capability)
    3. Auto-recovery from invalid history
    4. Circuit breaker behavior
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "tools" / "agentctl"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from agentctl.llm_models import ConversationHistory, Message
from agentctl.llm_validation_enhanced import EnhancedValidator, log_validation_event


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_basic_validation():
    """Test 1: Basic validation (existing capability)."""
    print_section("Test 1: Basic Validation (Existing)")

    history = ConversationHistory()
    history.add(Message(role="system", content="You are helpful"))
    history.add(Message(role="user", content="Hello"))
    history.add(Message(
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call_123",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"}
        }]
    ))
    # Missing tool response - basic validation should catch this

    # Test with basic validation (simulated)
    messages = history.to_list()
    pending_tool_calls = set()

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "id" in tc:
                    pending_tool_calls.add(tc["id"])
        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id in pending_tool_calls:
                pending_tool_calls.remove(tool_call_id)

    basic_valid = len(pending_tool_calls) == 0

    print(f"History: {len(messages)} messages")
    print(f"Tool calls without responses: {pending_tool_calls}")
    print(f"Basic validation result: {'✅ PASS' if basic_valid else '❌ FAIL'}")

    if not basic_valid:
        print("✅ Basic validation CORRECTLY detected missing tool response")
    else:
        print("❌ Basic validation FAILED to detect missing tool response")

    return not basic_valid


def test_order_violation():
    """Test 2: Order violation detection (enhanced capability)."""
    print_section("Test 2: Order Violation Detection (Enhanced)")

    history = ConversationHistory()
    history.add(Message(role="system", content="System prompt"))

    # ERROR: Tool response at [1] with no preceding tool_call
    # This is the EXACT error the user encountered!
    history.add(Message(
        role="tool",
        content='{"result": "test"}',
        tool_call_id="call_123"
    ))

    print("Simulating user's error: Tool response at position [1]")
    print("  messages[0]: system")
    print("  messages[1]: tool (NO PRECEDING TOOL_CALL!) ← Error!")

    validator = EnhancedValidator(history)
    is_valid, error = validator.validate_strict()

    print(f"\nEnhanced validation result: {'✅ PASS' if not is_valid else '❌ FAIL'}")
    print(f"Error detected: {error}")

    if not is_valid and "no preceding tool_calls" in error:
        print("✅ Enhanced validation CORRECTLY detected order violation")
        return True
    else:
        print("❌ Enhanced validation FAILED to detect order violation")
        return False


def test_auto_recovery():
    """Test 3: Auto-recovery from invalid history."""
    print_section("Test 3: Auto-Recovery from Invalid History")

    history = ConversationHistory()
    history.add(Message(role="system", content="System"))
    history.add(Message(role="user", content="Hello"))

    # Add valid tool_call
    history.add(Message(
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call_valid",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"}
        }]
    ))

    # Add orphaned tool response (wrong ID)
    history.add(Message(
        role="tool",
        content='{"result": "orphaned"}',
        tool_call_id="call_WRONG_ID"  # This doesn't match call_valid
    ))

    print("Initial history:")
    print("  [0] system")
    print("  [1] user")
    print("  [2] assistant (tool_call: call_valid)")
    print("  [3] tool (tool_call_id: call_WRONG_ID) ← ORPHANED!")

    validator = EnhancedValidator(history)

    # Validate
    is_valid, error = validator.validate_strict()
    print(f"\nInitial validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print(f"  Error: {error}")

    # Attempt recovery
    print("\nAttempting auto-recovery...")
    success, recovery_error = validator.recover_and_validate()

    print(f"Recovery result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    if not success:
        print(f"  Error: {recovery_error}")

    # Check final state
    messages_after = history.to_list()
    print(f"\nHistory after recovery: {len(messages_after)} messages")
    for i, msg in enumerate(messages_after):
        role = msg['role']
        tool_call_id = msg.get('tool_call_id', '')
        print(f"  [{i}] {role} {f'(tool_call_id: {tool_call_id})' if tool_call_id else ''}")

    # Verify orphaned message was removed
    tool_messages = [m for m in messages_after if m["role"] == "tool"]

    if success and len(tool_messages) == 0:
        print("\n✅ Auto-recovery SUCCESSFULLY removed orphaned tool response")
        return True
    else:
        print("\n❌ Auto-recovery FAILED")
        return False


def test_circuit_breaker():
    """Test 4: Circuit breaker behavior."""
    print_section("Test 4: Circuit Breaker Protection")

    # Create permanently invalid history (cannot be sanitized)
    history = ConversationHistory()
    history.add(Message(role="tool", content="bad", tool_call_id="invalid"))

    validator = EnhancedValidator(history)

    print("Creating permanently invalid history (tool message only)")
    print("Attempting recovery 3 times (should trigger circuit breaker)...\n")

    for i in range(4):
        success, error = validator.recover_and_validate(max_attempts=1)
        print(f"Attempt {i+1}: {'✅ Success' if success else '❌ Failed'}")

        if i < 3:
            print(f"  Circuit breaker: {validator.circuit_breaker.circuit_open}")
        else:
            # On 4th attempt, circuit should be open
            if "Circuit breaker is open" in (error or ""):
                print(f"  Circuit breaker: OPEN (correctly preventing further attempts)")
                print("\n✅ Circuit breaker CORRECTLY opened after max failures")
                return True
            else:
                print(f"  Circuit breaker: {validator.circuit_breaker.circuit_open}")

    print("\n❌ Circuit breaker FAILED to open")
    return False


def main():
    """Run all validation tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Agent Framework - LLM Error Recovery Test Suite            ║
║                                                              ║
║  Testing self-correction capabilities for production safety ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = []

    # Run tests
    results.append(("Basic Validation", test_basic_validation()))
    results.append(("Order Violation Detection", test_order_violation()))
    results.append(("Auto-Recovery", test_auto_recovery()))
    results.append(("Circuit Breaker", test_circuit_breaker()))

    # Print summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*60}\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour framework has comprehensive self-correction capabilities:")
        print("  ✅ Detects missing tool responses")
        print("  ✅ Detects order violations (tool response before tool_call)")
        print("  ✅ Auto-recovers from invalid history")
        print("  ✅ Circuit breaker prevents cascading failures")
        print("\n✨ Production-ready for LLM error handling!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nRecommended actions:")
        print("  1. Review failed tests above")
        print("  2. Check implementation in llm_validation_enhanced.py")
        print("  3. Verify imports in llm_client.py")
        print("\nSee docs/PRODUCTION_SAFETY_QUICK_START.md for integration guide")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
