"""
MCP Client for LangGraph Integration
Connects to MCP server and exposes tools as LangChain tools

IMPORTANT: Tool implementations are in agenticaicapstone/src/mcp/server.py
This file is a thin wrapper that:
1. Calls MCP server tools directly (no subprocess)
2. Formats JSON results as text for LLM consumption
3. Provides LangChain @tool decorators for LangGraph compatibility

To add a new tool:
1. Define in mcp/server.py with @mcp.tool()
2. Add wrapper here with @tool
3. Update get_mcp_tools() to include new tool
"""
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool


class MCPClient:
    """
    Client for connecting to MCP server and using its tools
    """
    
    def __init__(self, server_module: str = "agenticaicapstone.src.mcp.server"):
        """
        Initialize MCP client
        
        Args:
            server_module: Python module path to MCP server
        """
        self.server_module = server_module
        self.server_process = None
        self._tools_cache = None
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
        """
        # For FastMCP, we can import and call directly
        # This is more efficient than subprocess communication
        try:
            from agenticaicapstone.src.mcp.server import search_invoices, search_vendors, search_skus
            
            tool_map = {
                "search_invoices": search_invoices,
                "search_vendors": search_vendors,
                "search_skus": search_skus
            }
            
            if tool_name not in tool_map:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Call the MCP tool function directly
            result = tool_map[tool_name](**arguments)
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }


# Create global MCP client instance
_mcp_client = MCPClient()


@tool
def search_invoices_mcp(
    query: str,
    n_results: int = 5,
    filter_vendor: Optional[str] = None,
    filter_currency: Optional[str] = None,
    filter_language: Optional[str] = None
) -> str:
    """
    Search approved invoices using semantic search via MCP server. Use this when users ask about 
    invoices, purchases, payments, or specific transactions. Returns matching invoices with details
    like vendor, amount, date, and purchase order numbers.
    
    Args:
        query: Natural language search query (e.g., "safety equipment invoices", "high value purchases")
        n_results: Maximum number of results to return (default: 5, max: 20)
        filter_vendor: Optional filter by vendor name
        filter_currency: Optional filter by currency code (USD, EUR, etc.)
        filter_language: Optional filter by document language (English, German, etc.)
    
    Returns:
        JSON string with search results including invoice details and similarity scores
    """
    # Call MCP server tool
    result = _mcp_client.call_tool("search_invoices", {
        "query": query,
        "n_results": min(n_results, 20),
        "filter_vendor": filter_vendor,
        "filter_currency": filter_currency,
        "filter_language": filter_language
    })
    
    # Format results for LLM
    if not result.get("success"):
        return f"Search failed: {result.get('error', 'Unknown error')}"
    
    if result.get("n_results", 0) == 0:
        return f"No invoices found matching query: '{query}'"
    
    # Format results in a readable way
    results_text = f"Found {result['n_results']} invoice(s) for query: '{query}'\n\n"
    for item in result.get("results", []):
        metadata = item.get("metadata", {})
        results_text += f"Invoice ID: {item['invoice_id']}\n"
        results_text += f"  Source Invoice ID: {metadata.get('src_invoice_id', 'N/A')}\n"
        results_text += f"  Vendor: {metadata.get('vendor_name', 'N/A')}\n"
        results_text += f"  Amount: {metadata.get('amount', 'N/A')} {metadata.get('currency', '')}\n"
        results_text += f"  Date: {metadata.get('date', 'N/A')}\n"
        results_text += f"  PO Number: {metadata.get('po_number', 'N/A')}\n"
        results_text += f"  Language: {metadata.get('language', 'N/A')}\n"
        results_text += f"  Similarity Score: {item.get('similarity_score', 0):.2f}\n"
        results_text += f"  Summary: {item.get('content', '')[:200]}...\n\n"
    
    return results_text


@tool
def search_vendors_mcp(
    query: str,
    n_results: int = 5,
    filter_country: Optional[str] = None,
    filter_currency: Optional[str] = None
) -> str:
    """
    Search approved vendors using semantic search via MCP server. Use this when users ask about 
    suppliers, vendors, manufacturers, or company information. Returns vendor details including
    location, contact info, and business description.
    
    Args:
        query: Natural language search query (e.g., "German safety equipment suppliers", "vendors in USA")
        n_results: Maximum number of results to return (default: 5, max: 20)
        filter_country: Optional filter by country (e.g., "USA", "Germany")
        filter_currency: Optional filter by primary currency (USD, EUR, etc.)
    
    Returns:
        JSON string with vendor details and similarity scores
    """
    # Call MCP server tool
    result = _mcp_client.call_tool("search_vendors", {
        "query": query,
        "n_results": min(n_results, 20),
        "filter_country": filter_country,
        "filter_currency": filter_currency
    })
    
    if not result.get("success"):
        return f"Search failed: {result.get('error', 'Unknown error')}"
    
    if result.get("n_results", 0) == 0:
        return f"No vendors found matching query: '{query}'"
    
    # Format results
    results_text = f"Found {result['n_results']} vendor(s) for query: '{query}'\n\n"
    for item in result.get("results", []):
        metadata = item.get("metadata", {})
        results_text += f"Vendor ID: {item['vendor_id']}\n"
        results_text += f"  Source Invoice ID: {metadata.get('src_invoice_id', 'N/A')}\n"
        results_text += f"  Name: {metadata.get('vendor_name', 'N/A')}\n"
        results_text += f"  Country: {metadata.get('country', 'N/A')}\n"
        results_text += f"  Currency: {metadata.get('currency', 'N/A')}\n"
        results_text += f"  Address: {metadata.get('full_address', 'N/A')}\n"
        results_text += f"  Similarity Score: {item.get('similarity_score', 0):.2f}\n"
        results_text += f"  Description: {item.get('content', '')[:200]}...\n\n"
    
    return results_text


@tool
def search_skus_mcp(
    query: str,
    n_results: int = 5,
    filter_category: Optional[str] = None,
    filter_uom: Optional[str] = None
) -> str:
    """
    Search approved SKUs/products using semantic search via MCP server. Use this when users ask 
    about products, items, materials, or inventory. Returns SKU details including category,
    unit of measure, and GST rate.
    
    Args:
        query: Natural language search query (e.g., "safety helmets", "protective equipment", "construction materials")
        n_results: Maximum number of results to return (default: 5, max: 20)
        filter_category: Optional filter by product category (e.g., "Safety", "Electronics")
        filter_uom: Optional filter by unit of measure (e.g., "piece", "kg", "meter")
    
    Returns:
        JSON string with SKU details and similarity scores
    """
    # Call MCP server tool
    result = _mcp_client.call_tool("search_skus", {
        "query": query,
        "n_results": min(n_results, 20),
        "filter_category": filter_category,
        "filter_uom": filter_uom
    })
    
    if not result.get("success"):
        return f"Search failed: {result.get('error', 'Unknown error')}"
    
    if result.get("n_results", 0) == 0:
        return f"No SKUs found matching query: '{query}'"
    
    # Format results
    results_text = f"Found {result['n_results']} SKU(s) for query: '{query}'\n\n"
    for item in result.get("results", []):
        metadata = item.get("metadata", {})
        results_text += f"SKU ID: {item['item_code']}\n"
        results_text += f"  Source Invoice ID: {metadata.get('src_invoice_id', 'N/A')}\n"
        results_text += f"  Category: {metadata.get('category', 'N/A')}\n"
        results_text += f"  Unit of Measure: {metadata.get('uom', 'N/A')}\n"
        results_text += f"  GST Rate: {metadata.get('gst_rate', 'N/A')}%\n"
        results_text += f"  Similarity Score: {item.get('similarity_score', 0):.2f}\n"
        results_text += f"  Description: {item.get('content', '')[:200]}...\n\n"
    
    return results_text


def get_mcp_tools() -> List:
    """
    Get all MCP-backed tools for LangChain
    
    Returns:
        List of LangChain tools backed by MCP server
    """
    return [search_invoices_mcp, search_vendors_mcp, search_skus_mcp]