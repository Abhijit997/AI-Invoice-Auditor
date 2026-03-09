"""Pydantic models for API request/response"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List, Any


# ============================================================================
# Invoice Processing Models
# ============================================================================

class FileInfo(BaseModel):
    """Information about a file"""
    filename: str
    path: str
    size: int = Field(description="File size in bytes")
    created_at: datetime
    modified_at: datetime


class ProcessFileRequest(BaseModel):
    """Request to process a specific file"""
    filename: str = Field(description="Name of the file to process")


class ProcessFileResponse(BaseModel):
    """Response after processing a file"""
    success: bool
    message: str
    source_path: str
    destination_path: str
    processed_at: datetime


# ============================================================================
# Vector Store Models
# ============================================================================

class InvoiceDocument(BaseModel):
    """Model for adding invoice to vector store"""
    invoice_id: str = Field(..., description="Unique invoice identifier")
    content: str = Field(..., description="Invoice text content for embedding")
    metadata: Dict[str, Any] = Field(..., description="Invoice metadata")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "invoice_id": "INV-001",
                "content": "Invoice from Acme Corp for safety helmets, quantity 50, total $750.00",
                "metadata": {
                    "vendor_name": "Acme Corporation",
                    "vendor_id": "VEND-001",
                    "amount": 750.00,
                    "currency": "USD",
                    "date": "2026-02-27",
                    "po_number": "PO-2026-001",
                    "language": "en"
                }
            }
        }


class VendorDocument(BaseModel):
    """Model for adding vendor to vector store"""
    vendor_id: str = Field(..., description="Unique vendor identifier")
    content: str = Field(..., description="Vendor description for embedding")
    metadata: Dict[str, Any] = Field(..., description="Vendor metadata")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vendor_id": "VEND-001",
                "content": "Acme Corporation - leading supplier of safety equipment from USA",
                "metadata": {
                    "vendor_name": "Acme Corporation",
                    "country": "USA",
                    "currency": "USD",
                    "full_address": "123 Industrial Pkwy, Detroit, MI 48201, USA",
                    "src_invoice_id": "INV-001"
                }
            }
        }


class SKUDocument(BaseModel):
    """Model for adding SKU to vector store (matches sku_master.json structure)"""
    item_code: str = Field(..., description="Unique item code")
    content: str = Field(..., description="Item description for embedding")
    metadata: Dict[str, Any] = Field(..., description="SKU metadata from sku_master.json")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_code": "SKU-001",
                "content": "Safety category item - piece unit - 10% GST",
                "metadata": {
                    "category": "Safety",
                    "uom": "piece",
                    "gst_rate": 10,
                    "src_invoice_id": "INV-001"
                }
            }
        }


class SearchQuery(BaseModel):
    """Model for search queries"""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(5, ge=1, le=50, description="Number of results to return")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed query embedding (optional). If provided, must use same model as stored documents.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "safety helmets",
                "n_results": 5,
                "filter_dict": {"vendor_name": "Acme Corporation"}
            }
        }


class UpdateDocument(BaseModel):
    """Model for updating documents"""
    content: Optional[str] = Field(None, description="Updated content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    embedding: Optional[List[float]] = Field(None, description="Updated embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "amount": 800.00,
                    "status": "processed"
                }
            }
        }


class PODocument(BaseModel):
    """Model for adding Purchase Order to vector store (matches PO Records.json structure)"""
    po_number: str = Field(..., description="Unique PO number")
    content: str = Field(..., description="PO text content for embedding")
    metadata: Dict[str, Any] = Field(..., description="PO metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "po_number": "PO-1001",
                "content": "PO from Global Logistics Ltd for Pallet Wrapping Film (50 units), Industrial Gloves (120 units), Safety Helmets (30 units)",
                "metadata": {
                    "vendor_id": "VEND-001",
                    "line_items": [
                        {
                            "item_code": "SKU-001",
                            "description": "Pallet Wrapping Film",
                            "qty": 50,
                            "unit_price": 12.00,
                            "currency": "USD"
                        }
                    ],
                    "total_items": 3,
                    "total_quantity": 200,
                    "currency": "USD"
                }
            }
        }


class LineItem(BaseModel):
    """Model for PO line item (from PO Records.json)"""
    item_code: str
    description: str
    qty: int
    unit_price: float
    currency: str


# ============================================================================
# Staging Collection Models (with Review Tracking)
# ============================================================================

class ReviewMetadata(BaseModel):
    """Review tracking metadata for staging collections"""
    review_status: str = Field("pending", description="Review status: pending | approved | edited | rejected")
    reviewed_by: Optional[str] = Field(None, description="Username/email of reviewer")
    reviewed_at: Optional[str] = Field(None, description="Timestamp of review (ISO format)")
    review_notes: Optional[str] = Field(None, description="Optional review comments")
    original_values: Optional[str] = Field("{}", description="JSON string of original values before editing")


class InvoiceStagingDocument(BaseModel):
    """Model for invoice in staging collection with review tracking"""
    invoice_id: str = Field(..., description="Unique invoice identifier")
    content: str = Field(..., description="Invoice text content for embedding")
    metadata: Dict[str, Any] = Field(..., description="Invoice metadata with review tracking")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "invoice_id": "INV-001",
                "content": "Invoice from Acme Corp for safety helmets, quantity 50, total $750.00",
                "metadata": {
                    "vendor_name": "Acme Corporation",
                    "vendor_id": "VEND-001",
                    "amount": 750.00,
                    "currency": "USD",
                    "date": "2026-02-27",
                    "po_number": "PO-2026-001",
                    "language": "en",
                    "meta_file": "invoice_001.meta.json",
                    "processed_timestamp": "2026-02-28T10:00:00",
                    # Review tracking fields
                    "review_status": "pending",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "review_notes": None,
                    "original_values": "{}"  # JSON string
                }
            }
        }


class VendorStagingDocument(BaseModel):
    """Model for vendor in staging collection with review tracking"""
    vendor_id: str = Field(..., description="Unique vendor identifier")
    content: str = Field(..., description="Vendor description for embedding")
    metadata: Dict[str, Any] = Field(..., description="Vendor metadata with review tracking")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vendor_id": "VEND-001",
                "content": "Acme Corporation - leading supplier of safety equipment from USA",
                "metadata": {
                    "vendor_name": "Acme Corporation",
                    "country": "USA",
                    "currency": "USD",
                    "full_address": "123 Industrial Pkwy, Detroit, MI 48201, USA",
                    "src_invoice_id": "INV-001",
                    # Review tracking fields
                    "review_status": "pending",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "review_notes": None,
                    "original_values": "{}"  # JSON string
                }
            }
        }


class SKUStagingDocument(BaseModel):
    """Model for SKU in staging collection with review tracking"""
    item_code: str = Field(..., description="Unique item code")
    content: str = Field(..., description="Item description for embedding")
    metadata: Dict[str, Any] = Field(..., description="SKU metadata with review tracking")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_code": "SKU-001",
                "content": "Safety category item - piece unit - 10% GST",
                "metadata": {
                    "category": "Safety",
                    "uom": "piece",
                    "gst_rate": 10,
                    "src_invoice_id": "INV-001",
                    # Review tracking fields
                    "review_status": "pending",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "review_notes": None,
                    "original_values": "{}"  # JSON string
                }
            }
        }