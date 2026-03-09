"""
MCP Server for Invoice Auditor
Exposes semantic search tools for invoices, vendors, and SKUs from production collections.
"""
import httpx
from typing import Optional, Dict, Any
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("invoice-auditor")

# API Base URL
API_BASE_URL = "http://localhost:8000"


@mcp.tool()
def search_invoices(
    query: str,
    n_results: int = 5,
    filter_vendor: Optional[str] = None,
    filter_currency: Optional[str] = None,
    filter_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search invoices using semantic search across production invoice collection.
    
    Args:
        query: Natural language search query (e.g., "invoices for safety equipment")
        n_results: Maximum number of results to return (default: 5)
        filter_vendor: Optional vendor name filter
        filter_currency: Optional currency code filter (e.g., "USD", "EUR")
        filter_language: Optional language filter (e.g., "English", "German")
    
    Returns:
        Dictionary containing:
        - success: Whether search was successful
        - query: The original search query
        - n_results: Number of results returned
        - results: List of matching invoices with metadata and similarity scores
    
    Example:
        search_invoices("safety helmets from German vendors", n_results=3, filter_language="German")
    """
    # Build filter dictionary
    filter_dict = {}
    if filter_vendor:
        filter_dict["vendor_name"] = filter_vendor
    if filter_currency:
        filter_dict["currency"] = filter_currency
    if filter_language:
        filter_dict["language"] = filter_language
    
    # Make API request
    payload = {
        "query": query,
        "n_results": n_results
    }
    if filter_dict:
        payload["filter_dict"] = filter_dict
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_BASE_URL}/vector/invoices/search",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}",
            "query": query
        }


@mcp.tool()
def search_vendors(
    query: str,
    n_results: int = 5,
    filter_country: Optional[str] = None,
    filter_currency: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search vendors using semantic search across production vendor collection.
    
    Args:
        query: Natural language search query (e.g., "safety equipment suppliers in Europe")
        n_results: Maximum number of results to return (default: 5)
        filter_country: Optional country filter (e.g., "USA", "Germany")
        filter_currency: Optional currency filter (e.g., "USD", "EUR")
    
    Returns:
        Dictionary containing:
        - success: Whether search was successful
        - query: The original search query
        - n_results: Number of results returned
        - results: List of matching vendors with metadata and similarity scores
    
    Example:
        search_vendors("industrial suppliers", n_results=10, filter_country="USA")
    """
    # Build filter dictionary
    filter_dict = {}
    if filter_country:
        filter_dict["country"] = filter_country
    if filter_currency:
        filter_dict["currency"] = filter_currency
    
    # Make API request
    payload = {
        "query": query,
        "n_results": n_results
    }
    if filter_dict:
        payload["filter_dict"] = filter_dict
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_BASE_URL}/vector/vendors/search",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}",
            "query": query
        }


@mcp.tool()
def search_skus(
    query: str,
    n_results: int = 5,
    filter_category: Optional[str] = None,
    filter_uom: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search SKUs/products using semantic search across production SKU collection.
    
    Args:
        query: Natural language search query (e.g., "protective equipment for construction")
        n_results: Maximum number of results to return (default: 5)
        filter_category: Optional category filter (e.g., "Safety", "Electronics")
        filter_uom: Optional unit of measure filter (e.g., "piece", "kg", "meter")
    
    Returns:
        Dictionary containing:
        - success: Whether search was successful
        - query: The original search query
        - n_results: Number of results returned
        - results: List of matching SKUs with metadata and similarity scores
    
    Example:
        search_skus("helmets and harnesses", n_results=5, filter_category="Safety")
    """
    # Build filter dictionary
    filter_dict = {}
    if filter_category:
        filter_dict["category"] = filter_category
    if filter_uom:
        filter_dict["uom"] = filter_uom
    
    # Make API request
    payload = {
        "query": query,
        "n_results": n_results
    }
    if filter_dict:
        payload["filter_dict"] = filter_dict
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_BASE_URL}/vector/skus/search",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}",
            "query": query
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()