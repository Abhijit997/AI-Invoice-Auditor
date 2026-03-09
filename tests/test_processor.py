"""Test the invoice processing logic"""
import sys
from pathlib import Path

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent / "api"))

from processor import InvoiceProcessor, InvoiceMetadata

# Setup paths
BASE_DIR = Path(__file__).parent
INCOMING_FOLDER = BASE_DIR / "data" / "incoming"
PROCESSED_FOLDER = BASE_DIR / "data" / "processed"

print("=" * 60)
print("Testing Invoice Processor")
print("=" * 60)

# Initialize processor
processor = InvoiceProcessor(INCOMING_FOLDER, PROCESSED_FOLDER)

# Find meta files
print("\n1. Finding .meta.json files...")
meta_files = processor.find_meta_files()
print(f"   Found {len(meta_files)} metadata files:")
for mf in meta_files:
    print(f"   - {mf.name}")

# Validate invoices
print("\n2. Validating invoice pairs...")
pending_invoices = processor.get_pending_invoices()
print(f"   Total invoices: {len(pending_invoices)}")
for invoice in pending_invoices:
    status = "✓ VALID" if invoice['valid'] else "✗ INVALID"
    print(f"\n   {status}: {invoice['meta_file']}")
    print(f"      Sender: {invoice.get('sender', 'N/A')}")
    print(f"      Language: {invoice.get('language', 'N/A')}")
    print(f"      Expected attachments: {invoice.get('expected_attachments', [])}")
    print(f"      Found attachments: {invoice.get('found_attachments', [])}")
    if invoice.get('missing_files'):
        print(f"      ⚠ Missing: {invoice['missing_files']}")

# Test metadata parsing
if meta_files:
    print("\n3. Testing metadata parsing...")
    test_meta = InvoiceMetadata(meta_files[0])
    print(f"   File: {meta_files[0].name}")
    print(f"   Sender: {test_meta.sender}")
    print(f"   Subject: {test_meta.subject}")
    print(f"   Language: {test_meta.language}")
    print(f"   Received: {test_meta.received_timestamp}")
    print(f"   Attachments: {test_meta.attachments}")

print("\n" + "=" * 60)
print("✅ Invoice Processor Test Complete!")
print("=" * 60)

print("\n📋 Available API Endpoints:")
print("   GET  /invoices/pending         - List all pending invoices")
print("   POST /invoices/process/{file}  - Process specific invoice")
print("   POST /invoices/process-all     - Process all invoices")
print("   GET  /invoices/stats           - Get invoice statistics")