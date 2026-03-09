"""
Test script for MCP Server
Tests the three search tools by making direct API calls
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_invoice_search():
    """Test invoice search endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Search Invoices")
    print("="*60)
    
    payload = {
        "query": "safety equipment",
        "n_results": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/vector/invoices/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Success: Found {data.get('n_results', 0)} results")
        print(f"Query: {data.get('query')}")
        
        for result in data.get('results', [])[:2]:  # Show first 2
            print(f"\n  Rank {result['rank']}: {result['invoice_id']}")
            print(f"  Vendor: {result['metadata'].get('vendor_name')}")
            print(f"  Amount: {result['metadata'].get('amount')} {result['metadata'].get('currency')}")
            print(f"  Similarity: {result['similarity_score']:.3f}")
            
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def test_vendor_search():
    """Test vendor search endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Search Vendors")
    print("="*60)
    
    payload = {
        "query": "industrial suppliers",
        "n_results": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/vector/vendors/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Success: Found {data.get('n_results', 0)} results")
        print(f"Query: {data.get('query')}")
        
        for result in data.get('results', [])[:2]:  # Show first 2
            print(f"\n  Rank {result['rank']}: {result['vendor_id']}")
            print(f"  Name: {result['metadata'].get('vendor_name')}")
            print(f"  Country: {result['metadata'].get('country')}")
            print(f"  Similarity: {result['similarity_score']:.3f}")
            
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def test_sku_search():
    """Test SKU search endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Search SKUs")
    print("="*60)
    
    payload = {
        "query": "protective equipment",
        "n_results": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/vector/skus/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Success: Found {data.get('n_results', 0)} results")
        print(f"Query: {data.get('query')}")
        
        for result in data.get('results', [])[:2]:  # Show first 2
            print(f"\n  Rank {result['rank']}: {result['item_code']}")
            print(f"  Category: {result['metadata'].get('category')}")
            print(f"  UOM: {result['metadata'].get('uom')}")
            print(f"  Similarity: {result['similarity_score']:.3f}")
            
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def test_with_filters():
    """Test search with filters"""
    print("\n" + "="*60)
    print("TEST 4: Search with Filters")
    print("="*60)
    
    payload = {
        "query": "invoices",
        "n_results": 3,
        "filter_dict": {"currency": "USD"}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/vector/invoices/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Success: Found {data.get('n_results', 0)} USD invoices")
        
        for result in data.get('results', []):
            assert result['metadata']['currency'] == 'USD', "Filter not applied correctly!"
            print(f"  {result['invoice_id']}: {result['metadata']['currency']}")
            
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MCP Server API Tests")
    print("="*60)
    print(f"Testing endpoint: {BASE_URL}")
    print("Ensure FastAPI backend is running!")
    
    results = []
    results.append(("Invoice Search", test_invoice_search()))
    results.append(("Vendor Search", test_vendor_search()))
    results.append(("SKU Search", test_sku_search()))
    results.append(("Filtered Search", test_with_filters()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed}/{total}")