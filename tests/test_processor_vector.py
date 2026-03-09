"""
Test script for invoice processing with vector DB integration
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api.processor import InvoiceProcessor

def test_processor_with_vector_db():
    """Test invoice processor with vector DB loading"""
    
    print("=" * 70)
    print("INVOICE PROCESSOR TEST - Vector DB Integration")
    print("=" * 70)
    
    # Setup folders
    base_dir = Path(__file__).parent
    incoming_folder = base_dir / "data" / "incoming"
    processed_folder = base_dir / "data" / "processed"
    unprocessed_folder = base_dir / "data" / "unprocessed"
    
    # Initialize processor
    processor = InvoiceProcessor(incoming_folder, processed_folder, unprocessed_folder)
    
    print(f"\nFolders configured:")
    print(f"  Incoming: {incoming_folder}")
    print(f"  Processed: {processed_folder}")
    print(f"  Unprocessed: {unprocessed_folder}")
    
    # Find meta files
    meta_files = processor.find_meta_files()
    
    if not meta_files:
        print("\n⚠️  No invoice files found in incoming folder")
        print(f"   Please add .meta.json files to: {incoming_folder}")
        return
    
    print(f"\n📄 Found {len(meta_files)} invoice(s) to process")
    
    # Process each invoice
    for meta_file in meta_files:
        print(f"\n{'='*70}")
        print(f"Processing: {meta_file.name}")
        print('='*70)
        
        # Validate first
        validation = processor.validate_invoice_pair(meta_file)
        print(f"\nValidation:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Sender: {validation.get('sender', 'N/A')}")
        print(f"  Language: {validation.get('language', 'N/A')}")
        print(f"  Attachments: {validation.get('expected_attachments', [])}")
        
        if validation['valid']:
            # Process the invoice
            print(f"\n🔄 Processing invoice pair...")
            result = processor.process_invoice_pair(meta_file)
            
            print(f"\nResult:")
            print(f"  Success: {result['success']}")
            print(f"  Vector DB Loaded: {result.get('vector_db_loaded', 'N/A')}")
            
            if result.get('vector_db_error'):
                print(f"  Vector DB Error: {result['vector_db_error']}")
            
            print(f"  Files moved: {result.get('total_files_moved', 0)}")
            
            for file_info in result.get('moved_files', []):
                dest_folder = file_info.get('destination_folder', 'unknown')
                print(f"    - {file_info['source']} → {dest_folder}/{file_info['destination']}")
            
            if result.get('errors'):
                print(f"\n  ⚠️  Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
            
            if result.get('message'):
                print(f"\n  📝 {result['message']}")
        else:
            print(f"\n❌ Validation failed:")
            for error in validation.get('errors', []):
                print(f"  - {error}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n📌 Notes:")
    print("  - Files with successful vector DB loading → processed/")
    print("  - Files with failed vector DB loading → unprocessed/")
    print("  - Implement vector DB logic in load_invoice_to_vector_db()")


if __name__ == "__main__":
    print("\n🧪 Invoice Processor Test - Vector DB Integration\n")
    
    try:
        test_processor_with_vector_db()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()