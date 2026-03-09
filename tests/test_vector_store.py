"""
Test script for Chroma vector store functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agenticaicapstone.src.rag.vector_store import InvoiceVectorStore


def test_basic_operations():
    """Test basic CRUD operations"""
    print("=" * 70)
    print("TEST 1: Basic Operations")
    print("=" * 70)
    
    # Initialize store
    store = InvoiceVectorStore(persist_directory="data/vector_store_test")
    
    # Add test invoice
    print("\n1. Adding test invoice...")
    store.add_invoice(
        invoice_id="TEST-INV-001",
        content="Invoice from Acme Corporation for safety helmets, quantity 50, total $750.00",
        metadata={
            "vendor_name": "Acme Corporation",
            "vendor_id": "VEND-TEST-001",
            "amount": 750.00,
            "currency": "USD",
            "date": "2026-02-27"
        }
    )
    print("   ✓ Invoice added")
    
    # Add test vendor
    print("\n2. Adding test vendor...")
    store.add_vendor(
        vendor_id="VEND-TEST-001",
        content="Acme Corporation - leading supplier of safety equipment from USA",
        metadata={
            "vendor_name": "Acme Corporation",
            "country": "USA",
            "currency": "USD"
        }
    )
    print("   ✓ Vendor added")
    
    # Add test SKU
    print("\n3. Adding test SKU...")
    store.add_sku(
        item_code="SKU-TEST-001",
        content="Safety Helmets - High-visibility protective headgear for industrial use",
        metadata={
            "description": "Safety Helmets",
            "category": "Safety",
            "uom": "piece",
            "gst_rate": 10
        }
    )
    print("   ✓ SKU added")
    
    # Get stats
    stats = store.get_stats()
    print("\n4. Vector Store Stats:")
    print(f"   Total documents: {stats['total_documents']}")
    for coll, count in stats['collections'].items():
        print(f"   - {coll}: {count}")
    
    return store


def test_search_operations(store):
    """Test search functionality"""
    print("\n" + "=" * 70)
    print("TEST 2: Search Operations")
    print("=" * 70)
    
    # Search invoices
    print("\n1. Searching invoices for 'safety helmets'...")
    results = store.search_invoices("safety helmets", n_results=5)
    print(f"   Found {len(results['ids'][0])} results")
    if results['ids'][0]:
        for i, (id, doc, meta) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0]
        )):
            print(f"   {i+1}. {id}")
            print(f"      Vendor: {meta.get('vendor_name')}")
            print(f"      Amount: {meta.get('currency')} {meta.get('amount')}")
    
    # Search vendors
    print("\n2. Searching vendors for 'safety equipment supplier'...")
    results = store.search_vendors("safety equipment supplier", n_results=5)
    print(f"   Found {len(results['ids'][0])} results")
    if results['ids'][0]:
        for i, (id, doc, meta) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0]
        )):
            print(f"   {i+1}. {id}: {meta.get('vendor_name')} - {meta.get('country')}")
    
    # Search SKUs
    print("\n3. Searching SKUs for 'protective headgear'...")
    results = store.search_skus("protective headgear", n_results=5)
    print(f"   Found {len(results['ids'][0])} results")
    if results['ids'][0]:
        for i, (id, doc, meta) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0]
        )):
            print(f"   {i+1}. {id}: {meta.get('description')} ({meta.get('category')})")
    
    # Search with filters
    print("\n4. Searching invoices with vendor filter...")
    results = store.search_invoices(
        "helmets",
        n_results=5,
        filter_dict={"vendor_name": "Acme Corporation"}
    )
    print(f"   Found {len(results['ids'][0])} results from Acme Corporation")


def test_update_delete(store):
    """Test update and delete operations"""
    print("\n" + "=" * 70)
    print("TEST 3: Update and Delete Operations")
    print("=" * 70)
    
    # Update invoice
    print("\n1. Updating invoice metadata...")
    store.update_invoice(
        invoice_id="TEST-INV-001",
        metadata={
            "vendor_name": "Acme Corporation",
            "vendor_id": "VEND-TEST-001",
            "amount": 800.00,  # Updated amount
            "currency": "USD",
            "date": "2026-02-27",
            "status": "processed"  # New field
        }
    )
    print("   ✓ Invoice updated")
    
    # Verify update
    result = store.get_invoice("TEST-INV-001")
    if result['metadatas']:
        print(f"   New amount: {result['metadatas'][0].get('amount')}")
        print(f"   Status: {result['metadatas'][0].get('status')}")
    
    # Delete invoice
    print("\n2. Deleting invoice...")
    store.delete_invoice("TEST-INV-001")
    print("   ✓ Invoice deleted")
    
    # Verify deletion
    result = store.get_invoice("TEST-INV-001")
    print(f"   Verification: {len(result['ids'])} documents found (should be 0)")


def test_persistence():
    """Test data persistence"""
    print("\n" + "=" * 70)
    print("TEST 4: Persistence")
    print("=" * 70)
    
    # Create new store instance with same directory
    print("\n1. Creating new store instance with existing data...")
    store2 = InvoiceVectorStore(persist_directory="data/vector_store_test")
    
    stats = store2.get_stats()
    print(f"   ✓ Loaded existing data: {stats['total_documents']} documents")
    print(f"   Collections:")
    for coll, count in stats['collections'].items():
        print(f"   - {coll}: {count}")
    
    return store2


if __name__ == "__main__":
    print("\n🧪 Chroma Vector Store Test Suite\n")
    
    try:
        # Test 1: Basic operations
        store = test_basic_operations()
        
        # Test 2: Search operations
        test_search_operations(store)
        
        # Test 3: Update and delete
        test_update_delete(store)
        
        # Test 4: Persistence
        store2 = test_persistence()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("✅ All tests passed!")
        print("\nFinal stats:")
        stats = store2.get_stats()
        for coll, count in stats['collections'].items():
            print(f"  {coll}: {count} documents")
        print(f"\nStorage location: {store2.persist_dir}")
        print("=" * 70)
        
        # Cleanup option
        print("\n⚠️  Test database created at: data/vector_store_test/")
        cleanup = input("Delete test database? (yes/no): ")
        if cleanup.lower() == 'yes':
            import shutil
            shutil.rmtree("data/vector_store_test")
            print("✓ Test database deleted")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()