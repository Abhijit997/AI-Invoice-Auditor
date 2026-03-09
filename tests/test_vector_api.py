"""Test script for Vector Store API endpoints"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_vector_store_apis():
    """Test all vector store API endpoints"""
    
    print("=" * 70)
    print("VECTOR STORE API TESTS")
    print("=" * 70)
    
    # Test 1: Get vector store stats
    print("\n1. GET /vector/stats - Get vector store statistics")
    response = requests.get(f"{BASE_URL}/vector/stats")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total documents: {data['total_documents']}")
        print(f"   Collections: {data['collections']}")
    else:
        print(f"   Error: {response.text}")
    
    # Test 2: List collections
    print("\n2. GET /vector/collections - List all collections")
    response = requests.get(f"{BASE_URL}/vector/collections")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Collections: {data['collections']}")
    
    # Test 3: Add invoice to vector store
    print("\n3. POST /vector/invoices - Add invoice")
    invoice_data = {
        "invoice_id": "TEST-INV-001",
        "content": "Invoice from Acme Corporation for safety helmets, quantity 50, total $750.00",
        "metadata": {
            "vendor_name": "Acme Corporation",
            "vendor_id": "VEND-001",
            "amount": 750.00,
            "currency": "USD",
            "date": "2026-02-27",
            "language": "en"
        }
    }
    response = requests.post(f"{BASE_URL}/vector/invoices", json=invoice_data)
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        print(f"   Message: {data['message']}")
    else:
        print(f"   Error: {response.text}")
    
    # Test 4: Search invoices
    print("\n4. POST /vector/invoices/search - Search invoices")
    search_query = {
        "query": "safety helmets",
        "n_results": 5
    }
    response = requests.post(f"{BASE_URL}/vector/invoices/search", json=search_query)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found {data['n_results']} results")
        for result in data['results']:
            print(f"   - {result['invoice_id']}: {result['metadata'].get('vendor_name')} - ${result['metadata'].get('amount')}")
    
    # Test 5: Get invoice by ID
    print("\n5. GET /vector/invoices/TEST-INV-001 - Get specific invoice")
    response = requests.get(f"{BASE_URL}/vector/invoices/TEST-INV-001")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Invoice ID: {data['invoice_id']}")
        print(f"   Vendor: {data['metadata'].get('vendor_name')}")
    
    # Test 6: Update invoice
    print("\n6. PUT /vector/invoices/TEST-INV-001 - Update invoice")
    update_data = {
        "metadata": {
            "vendor_name": "Acme Corporation",
            "vendor_id": "VEND-001",
            "amount": 800.00,  # Updated amount
            "currency": "USD",
            "date": "2026-02-27",
            "language": "en",
            "status": "processed"  # New field
        }
    }
    response = requests.put(f"{BASE_URL}/vector/invoices/TEST-INV-001", json=update_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Message: {data['message']}")
    
    # Test 7: Add vendor
    print("\n7. POST /vector/vendors - Add vendor")
    vendor_data = {
        "vendor_id": "VEND-TEST-001",
        "content": "Acme Corporation - leading supplier of safety equipment from USA",
        "metadata": {
            "vendor_name": "Acme Corporation",
            "country": "USA",
            "currency": "USD"
        }
    }
    response = requests.post(f"{BASE_URL}/vector/vendors", json=vendor_data)
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        print(f"   Message: {data['message']}")
    
    # Test 8: Search vendors
    print("\n8. POST /vector/vendors/search - Search vendors")
    search_query = {
        "query": "safety equipment supplier",
        "n_results": 5
    }
    response = requests.post(f"{BASE_URL}/vector/vendors/search", json=search_query)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found {data['n_results']} results")
        for result in data['results']:
            print(f"   - {result['vendor_id']}: {result['metadata'].get('vendor_name')} ({result['metadata'].get('country')})")
    
    # Test 9: Add SKU
    print("\n9. POST /vector/skus - Add SKU")
    sku_data = {
        "item_code": "SKU-TEST-001",
        "content": "Safety Helmets - High-visibility protective headgear for industrial use - Safety category - piece unit - 10% GST",
        "metadata": {
            "category": "Safety",
            "uom": "piece",
            "gst_rate": 10
        }
    }
    response = requests.post(f"{BASE_URL}/vector/skus", json=sku_data)
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        print(f"   Message: {data['message']}")
    
    # Test 10: Search SKUs
    print("\n10. POST /vector/skus/search - Search SKUs")
    search_query = {
        "query": "protective headgear",
        "n_results": 5
    }
    response = requests.post(f"{BASE_URL}/vector/skus/search", json=search_query)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found {data['n_results']} results")
        for result in data['results']:
            print(f"   - {result['item_code']}: {result['metadata'].get('category')} - {result['metadata'].get('uom')} ({result['metadata'].get('gst_rate')}% GST)")
    
    # Test 11: Delete invoice
    print("\n11. DELETE /vector/invoices/TEST-INV-001 - Delete invoice")
    response = requests.delete(f"{BASE_URL}/vector/invoices/TEST-INV-001")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Message: {data['message']}")
    
    # Test 12: Final stats
    print("\n12. GET /vector/stats - Final statistics")
    response = requests.get(f"{BASE_URL}/vector/stats")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total documents: {data['total_documents']}")
        print(f"   Collections: {data['collections']}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🧪 Vector Store API Test Suite\n")
    print("⚠️  Make sure the API server is running:")
    print("   python -m uvicorn api.main:app --reload --port 8000\n")
    
    try:
        # Test if API is running
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API server is running\n")
            test_vector_store_apis()
        else:
            print("❌ API server returned unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Please start the server first with:")
        print("   python -m uvicorn api.main:app --reload --port 8000")