"""
Test script to verify file type validation in invoice processor
"""
from pathlib import Path
import json
import sys

# Add api folder to path
sys.path.insert(0, str(Path(__file__).parent))

from api.processor import InvoiceProcessor, ALLOWED_EXTENSIONS


def test_file_type_validation():
    """Test that only allowed file types are processed"""
    
    print("=" * 60)
    print("TESTING FILE TYPE VALIDATION")
    print("=" * 60)
    
    print(f"\nAllowed extensions: {sorted(ALLOWED_EXTENSIONS)}")
    
    # Test cases
    test_files = [
        ("invoice.pdf", True, "Valid PDF file"),
        ("invoice.docx", True, "Valid DOCX file"),
        ("invoice.json", True, "Valid JSON file"),
        ("invoice.csv", True, "Valid CSV file"),
        ("invoice.jpg", True, "Valid JPG file"),
        ("invoice.png", True, "Valid PNG file"),
        ("invoice.xlsx", False, "Invalid XLSX file"),
        ("invoice.txt", False, "Invalid TXT file"),
        ("invoice.zip", False, "Invalid ZIP file"),
        ("invoice.xml", False, "Invalid XML file"),
        ("invoice.html", False, "Invalid HTML file"),
    ]
    
    print("\n" + "-" * 60)
    print("FILE EXTENSION TESTS")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for filename, should_be_valid, description in test_files:
        ext = Path(filename).suffix.lower()
        is_valid = ext in ALLOWED_EXTENSIONS
        
        status = "✓ PASS" if is_valid == should_be_valid else "✗ FAIL"
        
        if is_valid == should_be_valid:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {description:25} | {filename:20} | Expected: {should_be_valid:5} | Got: {is_valid:5}")
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def test_validation_with_invalid_files():
    """Test validation logic with invalid file types"""
    
    print("\n" + "=" * 60)
    print("TESTING VALIDATION WITH INVALID FILE TYPES")
    print("=" * 60)
    
    # Setup test directories
    test_dir = Path("data/test_validation")
    incoming = test_dir / "incoming"
    processed = test_dir / "processed"
    
    # Create directories
    incoming.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    
    # Create test metadata with invalid file type
    meta_data = {
        "sender": "test@example.com",
        "subject": "Test Invoice with Invalid Type",
        "received_timestamp": "2026-02-27T10:00:00Z",
        "language": "en",
        "attachments": ["invoice.xlsx", "invoice.txt"]  # Invalid types
    }
    
    meta_file = incoming / "test_invalid.meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Create the invalid attachment files
    (incoming / "invoice.xlsx").write_text("fake excel data")
    (incoming / "invoice.txt").write_text("fake text data")
    
    # Test validation
    processor = InvoiceProcessor(incoming, processed)
    validation_result = processor.validate_invoice_pair(meta_file)
    
    print("\nValidation result:")
    print(f"  Valid: {validation_result['valid']}")
    print(f"  Meta file: {validation_result['meta_file']}")
    print(f"  Expected attachments: {validation_result['expected_attachments']}")
    print(f"  Invalid extensions: {validation_result.get('invalid_extensions', [])}")
    print(f"  Errors: {validation_result.get('errors', [])}")
    
    # Cleanup
    meta_file.unlink()
    (incoming / "invoice.xlsx").unlink()
    (incoming / "invoice.txt").unlink()
    
    # Check if validation correctly identified invalid files
    is_correct = (
        validation_result['valid'] == False and
        len(validation_result.get('invalid_extensions', [])) == 2
    )
    
    status = "✓ PASS" if is_correct else "✗ FAIL"
    print(f"\n{status} | Validation correctly rejected invalid file types")
    print("=" * 60)
    
    return is_correct


def test_validation_with_valid_files():
    """Test validation logic with valid file types"""
    
    print("\n" + "=" * 60)
    print("TESTING VALIDATION WITH VALID FILE TYPES")
    print("=" * 60)
    
    # Setup test directories
    test_dir = Path("data/test_validation")
    incoming = test_dir / "incoming"
    processed = test_dir / "processed"
    
    # Create directories
    incoming.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    
    # Create test metadata with valid file types
    meta_data = {
        "sender": "test@example.com",
        "subject": "Test Invoice with Valid Types",
        "received_timestamp": "2026-02-27T10:00:00Z",
        "language": "en",
        "attachments": ["invoice.pdf", "details.csv"]  # Valid types
    }
    
    meta_file = incoming / "test_valid.meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Create the valid attachment files
    (incoming / "invoice.pdf").write_text("fake pdf data")
    (incoming / "details.csv").write_text("fake csv data")
    
    # Test validation
    processor = InvoiceProcessor(incoming, processed)
    validation_result = processor.validate_invoice_pair(meta_file)
    
    print("\nValidation result:")
    print(f"  Valid: {validation_result['valid']}")
    print(f"  Meta file: {validation_result['meta_file']}")
    print(f"  Expected attachments: {validation_result['expected_attachments']}")
    print(f"  Found attachments: {validation_result['found_attachments']}")
    print(f"  Invalid extensions: {validation_result.get('invalid_extensions', [])}")
    print(f"  Errors: {validation_result.get('errors', [])}")
    
    # Cleanup
    meta_file.unlink()
    (incoming / "invoice.pdf").unlink()
    (incoming / "details.csv").unlink()
    
    # Check if validation correctly accepted valid files
    is_correct = (
        validation_result['valid'] == True and
        len(validation_result.get('invalid_extensions', [])) == 0 and
        len(validation_result['found_attachments']) == 2
    )
    
    status = "✓ PASS" if is_correct else "✗ FAIL"
    print(f"\n{status} | Validation correctly accepted valid file types")
    print("=" * 60)
    
    return is_correct


if __name__ == "__main__":
    print("\n🧪 File Type Validation Test Suite\n")
    
    results = []
    
    # Run tests
    results.append(("Extension Tests", test_file_type_validation()))
    results.append(("Invalid Files Test", test_validation_with_invalid_files()))
    results.append(("Valid Files Test", test_validation_with_valid_files()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    print("=" * 60)