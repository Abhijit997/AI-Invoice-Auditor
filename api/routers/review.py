"""Review endpoints for human-in-the-loop invoice validation"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import shutil
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agenticaicapstone.src.rag.vector_store import get_vector_store
    from agenticaicapstone.src.utils.azure_openai_client import AzureOpenAIClient
except ImportError:
    from ...agenticaicapstone.src.rag.vector_store import get_vector_store
    from ...agenticaicapstone.src.utils.azure_openai_client import AzureOpenAIClient

logger = logging.getLogger(__name__)

# Configure logging format to match standard: YYYY-MM-DD HH:MM:SS,mmm - module - LEVEL - message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

router = APIRouter(prefix="/review", tags=["Review"])

# Reuse vector store instance
_vector_store = None
_azure_client = None

def get_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store

def get_azure_client():
    """Get or create Azure OpenAI client for embeddings"""
    global _azure_client
    if _azure_client is None:
        try:
            _azure_client = AzureOpenAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
    return _azure_client


def generate_embedding(text: str) -> Optional[list]:
    """Generate embedding for text using Azure OpenAI"""
    client = get_azure_client()
    if client:
        try:
            embedding = client.create_embeddings(text)
            logger.info("Generated new embedding via Azure OpenAI")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
    return None


# ============================================================================
# Request/Response Models
# ============================================================================

class ReviewAction(BaseModel):
    """Request model for approve/reject actions"""
    reviewed_by: str = Field(..., description="Reviewer username or email")
    review_notes: Optional[str] = Field("", description="Optional review comments")
    # Edited metadata values (only changed fields)
    invoice_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated invoice metadata (changed fields only)")
    vendor_metadata: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Updated vendor metadata keyed by vendor_id")
    sku_metadata: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Updated SKU metadata keyed by item_code")
    # Edited content fields (triggers re-embedding)
    invoice_content: Optional[str] = Field(None, description="Updated invoice content text")
    vendor_content: Optional[Dict[str, str]] = Field(None, description="Updated vendor content keyed by vendor_id")
    sku_content: Optional[Dict[str, str]] = Field(None, description="Updated SKU content keyed by item_code")


# ============================================================================
# List Pending Reviews
# ============================================================================

@router.get("/pending")
async def list_pending_invoices():
    """Get all invoices in staging with review_status = pending"""
    try:
        store = get_store()
        # Get all from invoices_stage collection
        all_staged = store.invoices_stage_collection.get(
            where={"review_status": "pending"}
        )
        
        results = []
        if all_staged['ids']:
            for i, inv_id in enumerate(all_staged['ids']):
                meta = all_staged['metadatas'][i]
                results.append({
                    "invoice_id": inv_id,
                    "content": all_staged['documents'][i],
                    "metadata": meta,
                    "meta_file": meta.get("meta_file", ""),
                    "vendor_name": meta.get("vendor_name", ""),
                    "amount": meta.get("amount", 0),
                    "currency": meta.get("currency", ""),
                    "date": meta.get("date", "")
                })
        
        return {
            "success": True,
            "count": len(results),
            "invoices": results
        }
    except Exception as e:
        logger.error(f"Error listing pending invoices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/invoice/{invoice_id}")
async def get_invoice_review_details(invoice_id: str):
    """Get full invoice + vendors + SKUs combo for review"""
    try:
        store = get_store()
        
        # Get invoice from staging
        inv_result = store.get_invoice_stage(invoice_id)
        if not inv_result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged invoice {invoice_id} not found")
        
        invoice = {
            "invoice_id": inv_result['ids'][0],
            "content": inv_result['documents'][0],
            "metadata": inv_result['metadatas'][0]
        }
        
        # Get related vendors (by src_invoice_id)
        vendors = []
        try:
            vendor_results = store.vendors_stage_collection.get(
                where={"src_invoice_id": invoice_id}
            )
            if vendor_results['ids']:
                for i, vid in enumerate(vendor_results['ids']):
                    vendors.append({
                        "vendor_id": vid,
                        "content": vendor_results['documents'][i],
                        "metadata": vendor_results['metadatas'][i]
                    })
        except Exception as e:
            logger.warning(f"No vendors found for invoice {invoice_id}: {e}")
        
        # Get related SKUs (by src_invoice_id)
        skus = []
        try:
            sku_results = store.skus_stage_collection.get(
                where={"src_invoice_id": invoice_id}
            )
            if sku_results['ids']:
                for i, sid in enumerate(sku_results['ids']):
                    skus.append({
                        "item_code": sid,
                        "content": sku_results['documents'][i],
                        "metadata": sku_results['metadatas'][i]
                    })
        except Exception as e:
            logger.warning(f"No SKUs found for invoice {invoice_id}: {e}")
        
        # Determine the source file for preview
        meta_file = invoice['metadata'].get('meta_file', '')
        attachment_files = []
        if meta_file:
            # Read meta file to get attachment list
            processed_dir = Path("data/processed")
            meta_path = processed_dir / meta_file
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    attachment_files = meta_data.get('attachments', [])
                except Exception as e:
                    logger.warning(f"Could not read meta file {meta_path}: {e}")
        
        return {
            "success": True,
            "invoice": invoice,
            "vendors": vendors,
            "skus": skus,
            "meta_file": meta_file,
            "attachment_files": attachment_files
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting review details for {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Approve Invoice (promote to production)
# ============================================================================

@router.post("/invoice/{invoice_id}/approve")
async def approve_invoice(invoice_id: str, action: ReviewAction):
    """
    Approve invoice: copy to production collections, update staging status,
    and move files to data/accepted
    """
    try:
        store = get_store()
        now = datetime.now(timezone.utc).isoformat()
        
        # ---- Get staging data (including embeddings) ----
        inv_result = store.invoices_stage_collection.get(
            ids=[invoice_id],
            include=["documents", "metadatas", "embeddings"]
        )
        if not inv_result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged invoice {invoice_id} not found")
        
        inv_content = inv_result['documents'][0]
        inv_metadata = dict(inv_result['metadatas'][0])
        embeddings = inv_result.get('embeddings')
        inv_embedding = embeddings[0] if embeddings is not None and len(embeddings) > 0 else None
        
        # Track original values for any changes made
        original_values = {}
        
        # Apply edits to invoice metadata if provided
        if action.invoice_metadata:
            for key, new_val in action.invoice_metadata.items():
                if new_val is None:
                    # Remove field (user deleted it)
                    if key in inv_metadata:
                        original_values[f"invoice.{key}"] = inv_metadata[key]
                        del inv_metadata[key]
                elif key in inv_metadata:
                    # Update existing field
                    if inv_metadata[key] != new_val:
                        original_values[f"invoice.{key}"] = inv_metadata[key]
                        inv_metadata[key] = new_val
                else:
                    # Add new field (user added missing schema field)
                    original_values[f"invoice.{key}"] = None  # Track that it was added
                    inv_metadata[key] = new_val
        
        # Apply content edit and re-embed if invoice content changed
        if action.invoice_content and action.invoice_content.strip() != inv_content.strip():
            original_values["invoice.content"] = inv_content
            inv_content = action.invoice_content
            logger.info(f"Invoice {invoice_id} content changed by reviewer, calling Azure Embedding API to re-embed")
            inv_embedding = generate_embedding(inv_content)
            # Update staging with new content + embedding
            store.update_invoice_stage(
                invoice_id, content=inv_content, embedding=inv_embedding
            )
            logger.info(f"Updated invoice {invoice_id} content and re-embedded")
        
        # ---- Prepare production metadata (strip review fields) ----
        review_fields = ['review_status', 'reviewed_by', 'reviewed_at', 'review_notes', 'original_values']
        prod_inv_metadata = {k: v for k, v in inv_metadata.items() if k not in review_fields}
        
        # Add to production invoices collection (always with embedding)
        store.add_invoice(
            invoice_id=invoice_id,
            content=inv_content,
            metadata=prod_inv_metadata,
            embedding=inv_embedding
        )
        logger.info(f"Promoted invoice {invoice_id} to production")
        
        # ---- Process vendors ----
        vendor_results = store.vendors_stage_collection.get(
            where={"src_invoice_id": invoice_id},
            include=["documents", "metadatas", "embeddings"]
        )
        promoted_vendors = []
        if vendor_results['ids']:
            for i, vid in enumerate(vendor_results['ids']):
                v_metadata = dict(vendor_results['metadatas'][i])
                v_content = vendor_results['documents'][i]
                vendor_embeddings = vendor_results.get('embeddings')
                v_embedding = vendor_embeddings[i] if vendor_embeddings is not None and len(vendor_embeddings) > i else None
                
                # Apply vendor edits if provided
                if action.vendor_metadata and vid in action.vendor_metadata:
                    for key, new_val in action.vendor_metadata[vid].items():
                        if new_val is None:
                            # Remove field (user deleted it)
                            if key in v_metadata:
                                original_values[f"vendor.{vid}.{key}"] = v_metadata[key]
                                del v_metadata[key]
                        elif key in v_metadata:
                            # Update existing field
                            if v_metadata[key] != new_val:
                                original_values[f"vendor.{vid}.{key}"] = v_metadata[key]
                                v_metadata[key] = new_val
                        else:
                            # Add new field (user added missing schema field)
                            original_values[f"vendor.{vid}.{key}"] = None
                            v_metadata[key] = new_val
                
                # Apply vendor content edit and re-embed if changed
                if action.vendor_content and vid in action.vendor_content:
                    new_v_content = action.vendor_content[vid]
                    if new_v_content.strip() != v_content.strip():
                        original_values[f"vendor.{vid}.content"] = v_content
                        v_content = new_v_content
                        logger.info(f"Vendor {vid} content changed by reviewer, calling Azure Embedding API to re-embed")
                        v_embedding = generate_embedding(v_content)
                        store.update_vendor_stage(
                            vid, content=v_content, embedding=v_embedding
                        )
                        logger.info(f"Updated vendor {vid} content and re-embedded")
                
                # Strip review fields for production
                prod_v_metadata = {k: v for k, v in v_metadata.items() if k not in review_fields}
                
                # Add to production (always with embedding)
                store.add_vendor(
                    vendor_id=vid,
                    content=v_content,
                    metadata=prod_v_metadata,
                    embedding=v_embedding
                )
                
                # Update staging status
                v_metadata['review_status'] = 'approved'
                v_metadata['reviewed_by'] = action.reviewed_by
                v_metadata['reviewed_at'] = now
                v_metadata['review_notes'] = action.review_notes or ""
                v_metadata['original_values'] = json.dumps(
                    {k: v for k, v in original_values.items() if k.startswith(f"vendor.{vid}.")}
                ) if any(k.startswith(f"vendor.{vid}.") for k in original_values) else "{}"
                # Convert None to "" for ChromaDB
                v_metadata = {k: ("" if v is None else v) for k, v in v_metadata.items()}
                store.update_vendor_stage(vid, metadata=v_metadata)
                promoted_vendors.append(vid)
        
        # ---- Process SKUs ----
        sku_results = store.skus_stage_collection.get(
            where={"src_invoice_id": invoice_id},
            include=["documents", "metadatas", "embeddings"]
        )
        promoted_skus = []
        if sku_results['ids']:
            for i, sid in enumerate(sku_results['ids']):
                s_metadata = dict(sku_results['metadatas'][i])
                s_content = sku_results['documents'][i]
                sku_embeddings = sku_results.get('embeddings')
                s_embedding = sku_embeddings[i] if sku_embeddings is not None and len(sku_embeddings) > i else None
                
                # Apply SKU edits if provided
                if action.sku_metadata and sid in action.sku_metadata:
                    for key, new_val in action.sku_metadata[sid].items():
                        if new_val is None:
                            # Remove field (user deleted it)
                            if key in s_metadata:
                                original_values[f"sku.{sid}.{key}"] = s_metadata[key]
                                del s_metadata[key]
                        elif key in s_metadata:
                            # Update existing field
                            if s_metadata[key] != new_val:
                                original_values[f"sku.{sid}.{key}"] = s_metadata[key]
                                s_metadata[key] = new_val
                        else:
                            # Add new field (user added missing schema field)
                            original_values[f"sku.{sid}.{key}"] = None
                            s_metadata[key] = new_val
                
                # Apply SKU content edit and re-embed if changed
                if action.sku_content and sid in action.sku_content:
                    new_s_content = action.sku_content[sid]
                    if new_s_content.strip() != s_content.strip():
                        original_values[f"sku.{sid}.content"] = s_content
                        s_content = new_s_content
                        logger.info(f"SKU {sid} content changed by reviewer, calling Azure Embedding API to re-embed")
                        s_embedding = generate_embedding(s_content)
                        store.update_sku_stage(
                            sid, content=s_content, embedding=s_embedding
                        )
                        logger.info(f"Updated SKU {sid} content and re-embedded")
                
                # Strip review fields for production
                prod_s_metadata = {k: v for k, v in s_metadata.items() if k not in review_fields}
                
                # Add to production (always with embedding)
                store.add_sku(
                    item_code=sid,
                    content=s_content,
                    metadata=prod_s_metadata,
                    embedding=s_embedding
                )
                
                # Update staging status
                s_metadata['review_status'] = 'approved'
                s_metadata['reviewed_by'] = action.reviewed_by
                s_metadata['reviewed_at'] = now
                s_metadata['review_notes'] = action.review_notes or ""
                s_metadata['original_values'] = json.dumps(
                    {k: v for k, v in original_values.items() if k.startswith(f"sku.{sid}.")}
                ) if any(k.startswith(f"sku.{sid}.") for k in original_values) else "{}"
                s_metadata = {k: ("" if v is None else v) for k, v in s_metadata.items()}
                store.update_sku_stage(sid, metadata=s_metadata)
                promoted_skus.append(sid)
        
        # ---- Update invoice staging status ----
        inv_metadata['review_status'] = 'approved'
        inv_metadata['reviewed_by'] = action.reviewed_by
        inv_metadata['reviewed_at'] = now
        inv_metadata['review_notes'] = action.review_notes or ""
        inv_metadata['original_values'] = json.dumps(
            {k: v for k, v in original_values.items() if k.startswith("invoice.")}
        ) if any(k.startswith("invoice.") for k in original_values) else "{}"
        inv_metadata = {k: ("" if v is None else v) for k, v in inv_metadata.items()}
        store.update_invoice_stage(invoice_id, metadata=inv_metadata)
        
        # ---- Move files to data/accepted ----
        meta_file = inv_metadata.get('meta_file', '')
        moved_files = _move_files(meta_file, "data/accepted")
        
        return {
            "success": True,
            "message": f"Invoice {invoice_id} approved and promoted to production",
            "invoice_id": invoice_id,
            "promoted_vendors": promoted_vendors,
            "promoted_skus": promoted_skus,
            "moved_files": moved_files,
            "changes_made": original_values if original_values else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Reject Invoice
# ============================================================================

@router.post("/invoice/{invoice_id}/reject")
async def reject_invoice(invoice_id: str, action: ReviewAction):
    """
    Reject invoice: update staging status to rejected,
    move files to data/rejected
    """
    try:
        store = get_store()
        now = datetime.now(timezone.utc).isoformat()
        
        # Get invoice from staging
        inv_result = store.get_invoice_stage(invoice_id)
        if not inv_result['ids']:
            raise HTTPException(status_code=404, detail=f"Staged invoice {invoice_id} not found")
        
        inv_metadata = dict(inv_result['metadatas'][0])
        
        # Update invoice staging status
        inv_metadata['review_status'] = 'rejected'
        inv_metadata['reviewed_by'] = action.reviewed_by
        inv_metadata['reviewed_at'] = now
        inv_metadata['review_notes'] = action.review_notes or ""
        inv_metadata = {k: ("" if v is None else v) for k, v in inv_metadata.items()}
        store.update_invoice_stage(invoice_id, metadata=inv_metadata)
        
        # Update related vendors
        rejected_vendors = []
        try:
            vendor_results = store.vendors_stage_collection.get(
                where={"src_invoice_id": invoice_id}
            )
            if vendor_results['ids']:
                for i, vid in enumerate(vendor_results['ids']):
                    v_meta = dict(vendor_results['metadatas'][i])
                    v_meta['review_status'] = 'rejected'
                    v_meta['reviewed_by'] = action.reviewed_by
                    v_meta['reviewed_at'] = now
                    v_meta['review_notes'] = action.review_notes or ""
                    v_meta = {k: ("" if v is None else v) for k, v in v_meta.items()}
                    store.update_vendor_stage(vid, metadata=v_meta)
                    rejected_vendors.append(vid)
        except Exception as e:
            logger.warning(f"Error updating vendor staging for rejection: {e}")
        
        # Update related SKUs
        rejected_skus = []
        try:
            sku_results = store.skus_stage_collection.get(
                where={"src_invoice_id": invoice_id}
            )
            if sku_results['ids']:
                for i, sid in enumerate(sku_results['ids']):
                    s_meta = dict(sku_results['metadatas'][i])
                    s_meta['review_status'] = 'rejected'
                    s_meta['reviewed_by'] = action.reviewed_by
                    s_meta['reviewed_at'] = now
                    s_meta['review_notes'] = action.review_notes or ""
                    s_meta = {k: ("" if v is None else v) for k, v in s_meta.items()}
                    store.update_sku_stage(sid, metadata=s_meta)
                    rejected_skus.append(sid)
        except Exception as e:
            logger.warning(f"Error updating SKU staging for rejection: {e}")
        
        # Move files to data/rejected
        meta_file = inv_metadata.get('meta_file', '')
        moved_files = _move_files(meta_file, "data/rejected")
        
        return {
            "success": True,
            "message": f"Invoice {invoice_id} rejected",
            "invoice_id": invoice_id,
            "rejected_vendors": rejected_vendors,
            "rejected_skus": rejected_skus,
            "moved_files": moved_files
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# File Helper
# ============================================================================

@router.get("/file/{filename}")
async def get_file_for_preview(filename: str):
    """Get file info for preview - returns file path and type"""
    processed_dir = Path("data/processed")
    file_path = processed_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found in processed folder")
    
    return {
        "success": True,
        "filename": filename,
        "path": str(file_path.resolve()),
        "size": file_path.stat().st_size,
        "extension": file_path.suffix.lower()
    }


def _move_files(meta_file: str, destination_base: str) -> list:
    """Move meta file and its attachments to destination folder"""
    moved = []
    if not meta_file:
        return moved
    
    processed_dir = Path("data/processed")
    dest_dir = Path(destination_base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Read meta file for attachment list
    meta_path = processed_dir / meta_file
    attachment_files = []
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            attachment_files = meta_data.get('attachments', [])
        except Exception as e:
            logger.warning(f"Could not read meta file {meta_path}: {e}")
    
    # Move meta file
    if meta_path.exists():
        dest_path = dest_dir / meta_file
        shutil.move(str(meta_path), str(dest_path))
        moved.append(meta_file)
        logger.info(f"Moved {meta_file} to {destination_base}")
    
    # Move attachment files
    for att_file in attachment_files:
        att_path = processed_dir / att_file
        if att_path.exists():
            dest_path = dest_dir / att_file
            shutil.move(str(att_path), str(dest_path))
            moved.append(att_file)
            logger.info(f"Moved {att_file} to {destination_base}")
    
    return moved