"""Vector store endpoints for Chroma database operations"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict, Any
import sys
import logging
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agenticaicapstone.src.rag.vector_store import get_vector_store
    from api.models import (
        InvoiceDocument, VendorDocument, SKUDocument, 
        InvoiceStagingDocument, VendorStagingDocument, SKUStagingDocument,
        SearchQuery, UpdateDocument
    )
except ImportError:
    from ...agenticaicapstone.src.rag.vector_store import get_vector_store
    from ..models import (
        InvoiceDocument, VendorDocument, SKUDocument,
        InvoiceStagingDocument, VendorStagingDocument, SKUStagingDocument,
        SearchQuery, UpdateDocument
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector", tags=["Vector Store"])

# Initialize vector store (lazy loaded)
_vector_store = None

def get_store():
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store


def reset_store():
    """Reset the global vector store instance (used when ChromaDB has HNSW index errors)"""
    global _vector_store
    logger.warning("Resetting vector store due to index corruption")
    _vector_store = None
    return get_store()


def handle_chromadb_error(error: Exception, operation: str):
    """
    Handle ChromaDB errors, particularly HNSW index corruption after collection clearing
    
    Args:
        error: The exception that occurred
        operation: Description of the operation being performed
        
    Returns:
        True if error was handled and store was reset, False otherwise
    """
    error_msg = str(error).lower()
    
    # Check for HNSW segment reader errors (common after clearing collections)
    if "hnsw" in error_msg and ("nothing found on disk" in error_msg or "segment reader" in error_msg):
        logger.warning(f"Detected HNSW index corruption during {operation}, resetting store")
        reset_store()
        return True
    
    return False


# ============================================================================
# Collection Lock Management for Parallel Processing
# ============================================================================

# Global asyncio locks for each staging collection to prevent race conditions
_collection_locks: Dict[str, asyncio.Lock] = {
    "invoices_stage": asyncio.Lock(),
    "vendors_stage": asyncio.Lock(),
    "skus_stage": asyncio.Lock()
}

@router.post("/lock/invoices_stage")
async def lock_invoices_stage():
    """Lock invoices_stage collection for exclusive insert access"""
    try:
        await _collection_locks["invoices_stage"].acquire()
        return {"status": "locked", "collection": "invoices_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to lock collection: {str(e)}")

@router.post("/unlock/invoices_stage")
async def unlock_invoices_stage():
    """Unlock invoices_stage collection"""
    try:
        _collection_locks["invoices_stage"].release()
        return {"status": "unlocked", "collection": "invoices_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlock collection: {str(e)}")

@router.post("/lock/vendors_stage")
async def lock_vendors_stage():
    """Lock vendors_stage collection for exclusive insert access"""
    try:
        await _collection_locks["vendors_stage"].acquire()
        return {"status": "locked", "collection": "vendors_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to lock collection: {str(e)}")

@router.post("/unlock/vendors_stage")
async def unlock_vendors_stage():
    """Unlock vendors_stage collection"""
    try:
        _collection_locks["vendors_stage"].release()
        return {"status": "unlocked", "collection": "vendors_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlock collection: {str(e)}")

@router.post("/lock/skus_stage")
async def lock_skus_stage():
    """Lock skus_stage collection for exclusive insert access"""
    try:
        await _collection_locks["skus_stage"].acquire()
        return {"status": "locked", "collection": "skus_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to lock collection: {str(e)}")

@router.post("/unlock/skus_stage")
async def unlock_skus_stage():
    """Unlock skus_stage collection"""
    try:
        _collection_locks["skus_stage"].release()
        return {"status": "unlocked", "collection": "skus_stage"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlock collection: {str(e)}")


# ============================================================================
# Invoice Vector Operations
# ============================================================================

@router.post("/invoices", status_code=201)
async def add_invoice_to_vector_store(doc: InvoiceDocument):
    """Add invoice document to vector store"""
    try:
        store = get_store()
        store.add_invoice(doc.invoice_id, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"Invoice {doc.invoice_id} added to vector store",
            "invoice_id": doc.invoice_id
        }
    except Exception as e:
        logger.error(f"Error adding invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invoices/search")
async def search_invoices(query: SearchQuery):
    """Search invoices by semantic similarity"""
    try:
        logger.info(f"Invoice search request - Query: '{query.query}', n_results: {query.n_results}, Filters: {query.filter_dict}")
        store = get_store()
        results = store.search_invoices(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "invoice_id": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)  # Convert distance to similarity
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """Get specific invoice by ID"""
    try:
        store = get_store()
        result = store.get_invoice(invoice_id)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found")
        
        return {
            "success": True,
            "invoice_id": result['ids'][0],
            "content": result['documents'][0] if result['documents'] else None,
            "metadata": result['metadatas'][0] if result['metadatas'] else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/invoices/{invoice_id}")
async def update_invoice(invoice_id: str, update: UpdateDocument):
    """Update invoice content, metadata, or embedding"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_invoice(invoice_id)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found")
        
        store.update_invoice(invoice_id, update.content, update.metadata, update.embedding)
        return {
            "success": True,
            "message": f"Invoice {invoice_id} updated",
            "invoice_id": invoice_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invoices/{invoice_id}")
async def delete_invoice(invoice_id: str):
    """Delete invoice from vector store"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_invoice(invoice_id)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found")
        
        store.delete_invoice(invoice_id)
        return {
            "success": True,
            "message": f"Invoice {invoice_id} deleted",
            "invoice_id": invoice_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Vendor Vector Operations
# ============================================================================

@router.post("/vendors", status_code=201)
async def add_vendor_to_vector_store(doc: VendorDocument):
    """Add vendor to vector store"""
    try:
        store = get_store()
        store.add_vendor(doc.vendor_id, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"Vendor {doc.vendor_id} added to vector store",
            "vendor_id": doc.vendor_id
        }
    except Exception as e:
        logger.error(f"Error adding vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vendors/search")
async def search_vendors(query: SearchQuery):
    """Search vendors by semantic similarity"""
    try:
        logger.info(f"Vendor search request - Query: '{query.query}', n_results: {query.n_results}, Filters: {query.filter_dict}")
        store = get_store()
        results = store.search_vendors(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "vendor_id": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching vendors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    """Get specific vendor by ID"""
    try:
        store = get_store()
        result = store.get_vendor(vendor_id)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
        
        return {
            "success": True,
            "vendor_id": result['ids'][0],
            "content": result['documents'][0] if result['documents'] else None,
            "metadata": result['metadatas'][0] if result['metadatas'] else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vendors/{vendor_id}")
async def update_vendor(vendor_id: str, update: UpdateDocument):
    """Update vendor content, metadata, or embedding"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_vendor(vendor_id)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
        
        store.update_vendor(vendor_id, update.content, update.metadata, update.embedding)
        return {
            "success": True,
            "message": f"Vendor {vendor_id} updated",
            "vendor_id": vendor_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vendors/{vendor_id}")
async def delete_vendor(vendor_id: str):
    """Delete vendor from vector store"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_vendor(vendor_id)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
        
        store.delete_vendor(vendor_id)
        return {
            "success": True,
            "message": f"Vendor {vendor_id} deleted",
            "vendor_id": vendor_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SKU Vector Operations
# ============================================================================

@router.post("/skus", status_code=201)
async def add_sku_to_vector_store(doc: SKUDocument):
    """Add SKU/item to vector store"""
    try:
        store = get_store()
        store.add_sku(doc.item_code, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"SKU {doc.item_code} added to vector store",
            "item_code": doc.item_code
        }
    except Exception as e:
        logger.error(f"Error adding SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skus/search")
async def search_skus(query: SearchQuery):
    """Search SKUs by semantic similarity"""
    try:
        logger.info(f"SKU search request - Query: '{query.query}', n_results: {query.n_results}, Filters: {query.filter_dict}")
        store = get_store()
        results = store.search_skus(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "item_code": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching SKUs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skus/{item_code}")
async def get_sku(item_code: str):
    """Get specific SKU by item code"""
    try:
        store = get_store()
        result = store.get_sku(item_code)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"SKU {item_code} not found")
        
        return {
            "success": True,
            "item_code": result['ids'][0],
            "content": result['documents'][0] if result['documents'] else None,
            "metadata": result['metadatas'][0] if result['metadatas'] else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/skus/{item_code}")
async def update_sku(item_code: str, update: UpdateDocument):
    """Update SKU content, metadata, or embedding"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_sku(item_code)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"SKU {item_code} not found")
        
        store.update_sku(item_code, update.content, update.metadata, update.embedding)
        return {
            "success": True,
            "message": f"SKU {item_code} updated",
            "item_code": item_code
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/skus/{item_code}")
async def delete_sku(item_code: str):
    """Delete SKU from vector store"""
    try:
        store = get_store()
        
        # Check if exists
        existing = store.get_sku(item_code)
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"SKU {item_code} not found")
        
        store.delete_sku(item_code)
        return {
            "success": True,
            "message": f"SKU {item_code} deleted",
            "item_code": item_code
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Invoice Stage Vector Operations
# ============================================================================

@router.post("/invoices-stage", status_code=201)
async def add_invoice_to_stage(doc: InvoiceStagingDocument):
    """Add invoice document to staging collection with review tracking"""
    try:
        store = get_store()
        store.add_invoice_stage(doc.invoice_id, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"Invoice {doc.invoice_id} added to staging",
            "invoice_id": doc.invoice_id,
            "review_status": doc.metadata.get('review_status', 'pending')
        }
    except Exception as e:
        logger.error(f"Error adding invoice to stage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invoices-stage/search")
async def search_invoices_stage(query: SearchQuery):
    """Search staged invoices by semantic similarity"""
    try:
        store = get_store()
        results = store.search_invoices_stage(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "invoice_id": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching staged invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/invoices-stage/{invoice_id}")
async def get_invoice_stage(invoice_id: str):
    """Get specific staged invoice by ID"""
    try:
        store = get_store()
        result = store.get_invoice_stage(invoice_id)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged invoice {invoice_id} not found")
        
        return {
            "success": True,
            "invoice_id": result['ids'][0],
            "content": result['documents'][0],
            "metadata": result['metadatas'][0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting staged invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/invoices-stage/{invoice_id}")
async def update_invoice_stage(invoice_id: str, doc: UpdateDocument):
    """Update staged invoice"""
    try:
        store = get_store()
        store.update_invoice_stage(invoice_id, doc.content, doc.metadata, doc.embedding)
        
        return {
            "success": True,
            "message": f"Staged invoice {invoice_id} updated successfully",
            "invoice_id": invoice_id
        }
    except Exception as e:
        logger.error(f"Error updating staged invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invoices-stage/{invoice_id}")
async def delete_invoice_stage(invoice_id: str):
    """Delete staged invoice from vector store"""
    try:
        store = get_store()
        store.delete_invoice_stage(invoice_id)
        
        return {
            "success": True,
            "message": f"Staged invoice {invoice_id} deleted successfully",
            "invoice_id": invoice_id
        }
    except Exception as e:
        logger.error(f"Error deleting staged invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Vendor Stage Vector Operations
# ============================================================================

@router.post("/vendors-stage", status_code=201)
async def add_vendor_to_stage(doc: VendorStagingDocument):
    """Add vendor document to staging collection with review tracking"""
    try:
        store = get_store()
        store.add_vendor_stage(doc.vendor_id, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"Vendor {doc.vendor_id} added to staging",
            "vendor_id": doc.vendor_id,
            "review_status": doc.metadata.get('review_status', 'pending')
        }
    except Exception as e:
        logger.error(f"Error adding vendor to stage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vendors-stage/search")
async def search_vendors_stage(query: SearchQuery):
    """Search staged vendors by semantic similarity"""
    try:
        store = get_store()
        results = store.search_vendors_stage(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "vendor_id": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching staged vendors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vendors-stage/{vendor_id}")
async def get_vendor_stage(vendor_id: str):
    """Get specific staged vendor by ID"""
    try:
        store = get_store()
        result = store.get_vendor_stage(vendor_id)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged vendor {vendor_id} not found")
        
        return {
            "success": True,
            "vendor_id": result['ids'][0],
            "content": result['documents'][0],
            "metadata": result['metadatas'][0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting staged vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vendors-stage/{vendor_id}")
async def update_vendor_stage(vendor_id: str, doc: UpdateDocument):
    """Update staged vendor"""
    try:
        store = get_store()
        store.update_vendor_stage(vendor_id, doc.content, doc.metadata, doc.embedding)
        
        return {
            "success": True,
            "message": f"Staged vendor {vendor_id} updated successfully",
            "vendor_id": vendor_id
        }
    except Exception as e:
        logger.error(f"Error updating staged vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vendors-stage/{vendor_id}")
async def delete_vendor_stage(vendor_id: str):
    """Delete staged vendor from vector store"""
    try:
        store = get_store()
        store.delete_vendor_stage(vendor_id)
        
        return {
            "success": True,
            "message": f"Staged vendor {vendor_id} deleted successfully",
            "vendor_id": vendor_id
        }
    except Exception as e:
        logger.error(f"Error deleting staged vendor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SKU Stage Vector Operations
# ============================================================================

@router.post("/skus-stage", status_code=201)
async def add_sku_to_stage(doc: SKUStagingDocument):
    """Add SKU document to staging collection with review tracking"""
    try:
        store = get_store()
        store.add_sku_stage(doc.item_code, doc.content, doc.metadata, doc.embedding)
        return {
            "success": True,
            "message": f"SKU {doc.item_code} added to staging",
            "item_code": doc.item_code,
            "review_status": doc.metadata.get('review_status', 'pending')
        }
    except Exception as e:
        logger.error(f"Error adding SKU to stage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skus-stage/search")
async def search_skus_stage(query: SearchQuery):
    """Search staged SKUs by semantic similarity"""
    try:
        store = get_store()
        results = store.search_skus_stage(
            query.query,
            n_results=query.n_results,
            filter_dict=query.filter_dict,
            query_embedding=query.query_embedding
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "item_code": id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": float(1 - dist)
                })
        
        return {
            "success": True,
            "query": query.query,
            "n_results": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error searching staged SKUs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skus-stage/{item_code}")
async def get_sku_stage(item_code: str):
    """Get specific staged SKU by item code"""
    try:
        store = get_store()
        result = store.get_sku_stage(item_code)
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged SKU {item_code} not found")
        
        return {
            "success": True,
            "item_code": result['ids'][0],
            "content": result['documents'][0],
            "metadata": result['metadatas'][0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting staged SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/skus-stage/{item_code}")
async def update_sku_stage(item_code: str, doc: UpdateDocument):
    """Update staged SKU"""
    try:
        store = get_store()
        store.update_sku_stage(item_code, doc.content, doc.metadata, doc.embedding)
        
        return {
            "success": True,
            "message": f"Staged SKU {item_code} updated successfully",
            "item_code": item_code
        }
    except Exception as e:
        logger.error(f"Error updating staged SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/skus-stage/{item_code}")
async def delete_sku_stage(item_code: str):
    """Delete staged SKU from vector store"""
    try:
        store = get_store()
        store.delete_sku_stage(item_code)
        
        return {
            "success": True,
            "message": f"Staged SKU {item_code} deleted successfully",
            "item_code": item_code
        }
    except Exception as e:
        logger.error(f"Error deleting staged SKU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/stats")
async def get_vector_store_stats():
    """Get statistics about the vector store"""
    try:
        store = get_store()
        stats = store.get_stats()
        return {
            "success": True,
            "storage_location": stats['persist_directory'],
            "collections": stats['collections'],
            "total_documents": stats['total_documents']
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections():
    """List all collection names"""
    try:
        store = get_store()
        collections = store.list_collections()
        return {
            "success": True,
            "collections": collections
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-id/invoice")
async def get_next_invoice_id():
    """Get next available invoice ID
    
    Returns next sequential ID: INV-0000001
    """
    try:
        store = get_store()
        
        # Get all IDs from both collections
        prod_ids = store.invoices_collection.get()['ids']
        stage_ids = store.invoices_stage_collection.get()['ids']
        all_ids = prod_ids + stage_ids
        
        # Extract numeric parts and find max (only 7-digit sequential IDs)
        max_num = 0
        for id_str in all_ids:
            if id_str.startswith('INV-'):
                try:
                    num_part = id_str.replace('INV-', '')
                    # Only consider 7-digit sequential IDs, ignore timestamp-based IDs
                    if len(num_part) == 7 and num_part.isdigit():
                        num = int(num_part)
                        max_num = max(max_num, num)
                except ValueError:
                    continue
        
        # Generate next ID
        next_id = f"INV-{max_num + 1:07d}"
        
        return {
            "success": True,
            "id": next_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating invoice ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-id/vendor")
async def get_next_vendor_id():
    """Get next available vendor ID
    
    Returns next sequential ID: VND-0000001
    """
    try:
        store = get_store()
        
        # Get all IDs from both collections
        prod_ids = store.vendors_collection.get()['ids']
        stage_ids = store.vendors_stage_collection.get()['ids']
        all_ids = prod_ids + stage_ids
        
        # Extract numeric parts and find max (only 7-digit sequential IDs)
        max_num = 0
        for id_str in all_ids:
            if id_str.startswith('VND-'):
                try:
                    num_part = id_str.replace('VND-', '')
                    # Only consider 7-digit sequential IDs, ignore timestamp-based IDs
                    if len(num_part) == 7 and num_part.isdigit():
                        num = int(num_part)
                        max_num = max(max_num, num)
                except ValueError:
                    continue
        
        # Generate next ID
        next_id = f"VND-{max_num + 1:07d}"
        
        return {
            "success": True,
            "id": next_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating vendor ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-id/sku")
async def get_next_sku_id():
    """Get next available SKU ID
    
    Returns next sequential ID: SKU-0000001
    """
    try:
        store = get_store()
        
        # Get all IDs from both collections
        prod_ids = store.skus_collection.get()['ids']
        stage_ids = store.skus_stage_collection.get()['ids']
        all_ids = prod_ids + stage_ids
        
        # Extract numeric parts and find max (only 7-digit sequential IDs)
        max_num = 0
        for id_str in all_ids:
            if id_str.startswith('SKU-'):
                try:
                    num_part = id_str.replace('SKU-', '')
                    # Only consider 7-digit sequential IDs, ignore timestamp-based IDs
                    if len(num_part) == 7 and num_part.isdigit():
                        num = int(num_part)
                        max_num = max(max_num, num)
                except ValueError:
                    continue
        
        # Generate next ID
        next_id = f"SKU-{max_num + 1:07d}"
        
        return {
            "success": True,
            "id": next_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SKU ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/staging")
async def clear_staging_collections():
    """Delete all data from all staging collections (WARNING: Deletes all staging data!)
    
    This clears all data from:
    - invoices_stage
    - vendors_stage  
    - skus_stage
    
    The collections remain but are emptied.
    """
    try:
        store = get_store()
        
        # Clear all staging collections using efficient method
        cleared_counts = store.clear_all_staging_data()
        
        # Reset store to avoid HNSW index corruption issues
        reset_store()
        logger.info("Vector store reset after clearing staging collections")
        
        return {
            "success": True,
            "message": "All staging collections have been cleared",
            "collections_cleared": list(cleared_counts.keys()),
            "documents_deleted": cleared_counts,
            "total_deleted": sum(cleared_counts.values())
        }
    except Exception as e:
        logger.error(f"Error clearing staging collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/production")
async def clear_production_collections():
    """Delete all data from all production collections (WARNING: Deletes all production data!)
    
    This clears all data from:
    - invoices
    - vendors
    - skus
    
    The collections remain but are emptied.
    """
    try:
        store = get_store()
        
        # Clear all production collections using efficient method
        cleared_counts = store.clear_all_production_data()
        
        # Reset store to avoid HNSW index corruption issues
        reset_store()
        logger.info("Vector store reset after clearing production collections")
        
        return {
            "success": True,
            "message": "All production collections have been cleared",
            "collections_cleared": list(cleared_counts.keys()),
            "documents_deleted": cleared_counts,
            "total_deleted": sum(cleared_counts.values())
        }
    except Exception as e:
        logger.error(f"Error clearing production collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))