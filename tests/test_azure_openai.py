"""
Test script for Azure OpenAI Client

This script demonstrates various usage patterns and tests the Azure OpenAI integration.

Before running:
1. Create .env file with Azure credentials
2. Install dependencies: pip install -r requirements.txt
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agenticaicapstone.src.utils.azure_openai_client import (
    AzureOpenAIClient,
    send_prompt,
    send_prompt_with_image
)


def test_basic_prompt():
    """Test 1: Basic text prompt"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Text Prompt")
    print("=" * 70)
    
    try:
        client = AzureOpenAIClient()
        response = client.chat("What is 2 + 2? Answer with just the number.")
        print(f"✓ Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_structured_extraction():
    """Test 2: Structured data extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: Structured Data Extraction")
    print("=" * 70)
    
    invoice_text = """
    INVOICE
    Invoice Number: INV-2026-001
    Date: February 27, 2026
    Vendor: Acme Corporation
    
    Items:
    1. Widget A - Qty: 10 - Price: $50.00 - Total: $500.00
    2. Widget B - Qty: 5 - Price: $75.00 - Total: $375.00
    
    Subtotal: $875.00
    Tax (10%): $87.50
    Total: $962.50
    """
    
    prompt = f"""Extract invoice information from the following text and return as JSON:

{invoice_text}

Return JSON with these fields:
- invoice_date (YYYY-MM-DD format)
- vendor_name
- line_items (array with description, quantity, unit_price, total)
- subtotal
- tax
- total

Return ONLY valid JSON, no other text."""

    try:
        client = AzureOpenAIClient()
        response = client.chat(
            prompt=prompt,
            system_message="You are an invoice data extraction expert. Always return valid JSON.",
            temperature=0.0
        )
        
        # Validate it's valid JSON
        data = json.loads(response)
        print(f"✓ Extracted data:")
        print(json.dumps(data, indent=2))
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {e}")
        print(f"Response was: {response}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_image_analysis():
    """Test 3: Image analysis with vision model"""
    print("\n" + "=" * 70)
    print("TEST 3: Image Analysis")
    print("=" * 70)
    
    # Check for test images
    test_images = [
        Path("data/incoming/INV_EN_005_scan.png"),
        Path("data/incoming/INV_EN_001.pdf"),
    ]
    
    image_found = None
    for img_path in test_images:
        if img_path.exists() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_found = img_path
            break
    
    if not image_found:
        print("⊘ Skipped: No test image found in data/incoming/")
        print(f"  Looked for: {[str(p) for p in test_images]}")
        return None
    
    try:
        client = AzureOpenAIClient()
        response = client.chat_with_image(
            prompt="Describe what you see in this image. Is it an invoice or document? What information can you extract?",
            image_path=image_found,
            temperature=0.3
        )
        print(f"✓ Image: {image_found.name}")
        print(f"✓ Analysis: {response}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_invoice_extraction_from_image():
    """Test 4: Complete invoice extraction from image"""
    print("\n" + "=" * 70)
    print("TEST 4: Invoice Data Extraction from Image")
    print("=" * 70)
    
    # Look for PNG invoice
    test_image = Path("data/incoming/INV_EN_005_scan.png")
    
    if not test_image.exists():
        print("⊘ Skipped: Test image not found")
        return None
    
    prompt = """Extract invoice information from this image and return as JSON:

{
    "invoice_date": "YYYY-MM-DD",
    "vendor_name": "string",
    "total_amount": number,
    "currency": "string",
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "total": number
        }
    ]
}

Return ONLY valid JSON, no other text."""

    try:
        client = AzureOpenAIClient()
        response = client.chat_with_image(
            prompt=prompt,
            image_path=test_image,
            system_message="You are an invoice extraction expert. Always return valid JSON.",
            temperature=0.0
        )
        
        # Validate JSON
        data = json.loads(response)
        print(f"✓ Extracted from {test_image.name}:")
        print(json.dumps(data, indent=2))
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {e}")
        print(f"Response was: {response}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multi_turn_conversation():
    """Test 5: Multi-turn conversation"""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Turn Conversation")
    print("=" * 70)
    
    try:
        client = AzureOpenAIClient()
        
        conversation = [
            {"role": "system", "content": "You are a helpful invoice processing assistant."},
            {"role": "user", "content": "I have an invoice from ABC Company for $1,500"},
            {"role": "assistant", "content": "I can help you process that invoice. What information do you need?"},
            {"role": "user", "content": "Is this amount unusually high?"}
        ]
        
        response = client.chat_conversation(conversation, temperature=0.7)
        print(f"✓ Conversation response: {response}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ocr_extraction():
    """Test 6: OCR text extraction from image"""
    print("\n" + "=" * 70)
    print("TEST 6: OCR Text Extraction")
    print("=" * 70)
    
    test_image = Path("data/incoming/INV_EN_005_scan.png")
    
    if not test_image.exists():
        print("⊘ Skipped: Test image not found")
        return None
    
    try:
        cv_client = AzureComputerVisionClient()
        extracted_text = cv_client.extract_text(test_image)
        print(f"✓ Extracted text from {test_image.name}:")
        print("-" * 70)
        print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        print("-" * 70)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_convenience_functions():
    """Test 7: Quick convenience functions"""
    print("\n" + "=" * 70)
    print("TEST 7: Convenience Functions")
    print("=" * 70)
    
    try:
        # Test send_prompt
        response = send_prompt("What is the purpose of an invoice? Answer in one sentence.")
        print(f"✓ send_prompt: {response}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_environment():
    """Check if .env file is properly configured"""
    print("\n" + "=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    optional_vars = [
        "AZURE_CV_ENDPOINT",
        "AZURE_CV_KEY"
    ]
    
    all_ok = True
    
    print("\nRequired variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  ✗ {var}: NOT SET")
            all_ok = False
    
    print("\nOptional variables (for OCR):")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  ⊘ {var}: NOT SET (OCR tests will be skipped)")
    
    if not all_ok:
        print("\n⚠️  Missing required environment variables!")
        print("   Create a .env file with your Azure credentials.")
        print("   See api/AZURE_OPENAI_SETUP.md for details.")
    
    return all_ok


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" Azure OpenAI Client - Test Suite")
    print("=" * 70)
    
    # Check environment first
    env_ok = check_environment()
    
    if not env_ok:
        print("\n❌ Environment not configured properly. Please set up .env file first.")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Basic Prompt", test_basic_prompt),
        ("Structured Extraction", test_structured_extraction),
        ("Image Analysis", test_image_analysis),
        ("Invoice from Image", test_invoice_extraction_from_image),
        ("Multi-Turn Conversation", test_multi_turn_conversation),
        ("OCR Extraction", test_ocr_extraction),
        ("Convenience Functions", test_convenience_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"  {status} | {test_name}")
    
    print("-" * 70)
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 70)
    
    if failed > 0:
        print("\n❌ Some tests failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)