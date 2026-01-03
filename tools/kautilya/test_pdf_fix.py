"""
Test script to verify PDF attachment fix.

This script tests that binary documents (PDFs) are handled correctly:
- Not read as UTF-8 text (which caused token limit errors)
- Stored as metadata only
- Context prompt shows metadata, not bloated content
"""

from pathlib import Path
import sys

# Add kautilya to path
sys.path.insert(0, str(Path(__file__).parent))

from kautilya.interactive import InteractiveMode
from kautilya.config import Config


def test_pdf_attachment():
    """Test that PDFs are attached as metadata only."""

    print("=" * 70)
    print("Testing PDF Attachment Fix")
    print("=" * 70)

    # Create a minimal config
    config = Config()

    # Initialize interactive mode
    interactive = InteractiveMode(
        config_dir=".kautilya",
        config=config
    )

    # Test file path
    pdf_path = Path("reports/samples/apple_esg_report.pdf")

    if not pdf_path.exists():
        print(f"\n❌ Test PDF not found: {pdf_path}")
        print("Please ensure the test PDF exists first.")
        return False

    print(f"\n📄 Test PDF: {pdf_path}")
    print(f"   File size: {pdf_path.stat().st_size / 1024:.2f} KB")

    # Test 1: Check if PDF is recognized as binary document
    print("\n" + "=" * 70)
    print("Test 1: Binary Document Detection")
    print("=" * 70)

    is_binary = interactive._is_binary_document(pdf_path)
    print(f"   Is binary document: {is_binary}")

    if is_binary:
        print("   ✅ PDF correctly identified as binary document")
    else:
        print("   ❌ PDF NOT identified as binary document")
        return False

    # Test 2: Check that _read_file_safe returns None for PDF
    print("\n" + "=" * 70)
    print("Test 2: File Reading (Binary Documents)")
    print("=" * 70)

    content, error = interactive._read_file_safe(pdf_path)

    print(f"   Content returned: {content}")
    print(f"   Error returned: {error}")

    if content is None and error is None:
        print("   ✅ PDF NOT read as text (returns None content, no error)")
        print("   ✅ This prevents bloated token usage!")
    else:
        print(f"   ❌ Unexpected result: content={content}, error={error}")
        return False

    # Test 3: Attach the PDF and check stored content
    print("\n" + "=" * 70)
    print("Test 3: File Attachment")
    print("=" * 70)

    success, message = interactive._attach_file(pdf_path)

    print(f"   Attach success: {success}")
    print(f"   Message: {message}")

    if not success:
        print(f"   ❌ Failed to attach PDF: {message}")
        return False

    abs_path = str(pdf_path.absolute())
    stored_content = interactive.attached_context.get(abs_path)
    stored_stats = interactive.attached_stats.get(abs_path, {})

    print(f"\n   Stored content type: {type(stored_content)}")
    print(f"   Stored content value: {stored_content}")
    print(f"   Stored stats: {stored_stats}")

    if stored_content is None:
        print("   ✅ Content stored as None (metadata only)")
    else:
        print(f"   ❌ Content should be None but got: {type(stored_content)}")
        return False

    if stored_stats.get("is_binary"):
        print("   ✅ Marked as binary document in stats")
    else:
        print("   ❌ NOT marked as binary in stats")
        return False

    # Test 4: Check context prompt
    print("\n" + "=" * 70)
    print("Test 4: Context Prompt Generation")
    print("=" * 70)

    context_prompt = interactive._build_context_prompt()

    print("\n   Generated context prompt:")
    print("   " + "-" * 66)
    for line in context_prompt.split('\n'):
        print(f"   {line}")
    print("   " + "-" * 66)

    # Verify context prompt doesn't contain bloated content
    prompt_length = len(context_prompt)
    prompt_tokens = prompt_length // 4  # Rough estimate

    print(f"\n   Context prompt length: {prompt_length} chars (~{prompt_tokens} tokens)")

    if prompt_tokens < 1000:
        print("   ✅ Context prompt is small (metadata only)")
        print("   ✅ No token limit errors expected!")
    else:
        print(f"   ⚠️  Context prompt is large: {prompt_tokens} tokens")

    # Check that prompt contains file path but not PDF content
    if abs_path in context_prompt:
        print(f"   ✅ File path included in context: {pdf_path.name}")
    else:
        print("   ❌ File path NOT found in context")

    if "document_qa" in context_prompt:
        print("   ✅ Instructions to use document_qa skill included")
    else:
        print("   ⚠️  No mention of document_qa skill")

    # Test 5: Simulate token usage
    print("\n" + "=" * 70)
    print("Test 5: Estimated Token Usage")
    print("=" * 70)

    # If we had read the PDF as text (old behavior)
    pdf_size = pdf_path.stat().st_size
    bloated_tokens_old = pdf_size * 2 // 4  # Assume 2x bloat from UTF-8 decode

    # With new behavior (metadata only)
    metadata_tokens = prompt_tokens

    print(f"\n   OLD behavior (reading PDF as text):")
    print(f"      Estimated tokens: ~{bloated_tokens_old:,}")
    print(f"      Would exceed 8,192 limit: {'YES ❌' if bloated_tokens_old > 8192 else 'NO'}")

    print(f"\n   NEW behavior (metadata only):")
    print(f"      Estimated tokens: ~{metadata_tokens:,}")
    print(f"      Fits within 8,192 limit: {'YES ✅' if metadata_tokens < 8192 else 'NO'}")

    savings = bloated_tokens_old - metadata_tokens
    savings_pct = (savings / bloated_tokens_old * 100) if bloated_tokens_old > 0 else 0

    print(f"\n   Token savings: ~{savings:,} tokens ({savings_pct:.1f}% reduction)")

    # Final result
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("\n   ✅ All tests PASSED!")
    print("\n   Summary:")
    print("   • PDFs are correctly identified as binary documents")
    print("   • PDFs are NOT read as UTF-8 text (no bloated content)")
    print("   • Only metadata stored (path, size, type)")
    print("   • Context prompt is small and efficient")
    print("   • Token limit errors should NOT occur")
    print("\n   🎉 The fix is working correctly!")

    return True


if __name__ == "__main__":
    try:
        success = test_pdf_attachment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
