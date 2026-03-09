"""Invoice processing endpoints"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import logging
import sys

# Add parent directory to path for processor import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.processor import InvoiceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invoices", tags=["Invoice Processing"])

# Configure paths
BASE_DIR = Path(__file__).parent.parent.parent
INCOMING_FOLDER = BASE_DIR / "data" / "incoming"
PROCESSED_FOLDER = BASE_DIR / "data" / "processed"
UNPROCESSED_FOLDER = BASE_DIR / "data" / "unprocessed"

# Ensure directories exist
INCOMING_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
UNPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize invoice processor
invoice_processor = InvoiceProcessor(INCOMING_FOLDER, PROCESSED_FOLDER, UNPROCESSED_FOLDER)


@router.get("/pending")
async def get_pending_invoices():
    """
    Get list of pending invoices with validation status
    Returns invoice metadata and checks if attachments exist
    """
    try:
        pending_invoices = invoice_processor.get_pending_invoices()
        
        valid_count = sum(1 for inv in pending_invoices if inv.get('valid', False))
        invalid_count = len(pending_invoices) - valid_count
        
        return {
            "success": True,
            "total_invoices": len(pending_invoices),
            "valid_invoices": valid_count,
            "invalid_invoices": invalid_count,
            "invoices": pending_invoices
        }
    except Exception as e:
        logger.error(f"Error getting pending invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/{meta_filename}")
async def process_single_invoice(meta_filename: str):
    """
    Process a single invoice pair (metadata + attachments)
    Only processes if corresponding attachment files exist
    """
    try:
        if not meta_filename.endswith('.meta.json'):
            raise HTTPException(
                status_code=400,
                detail="Filename must end with .meta.json"
            )
        
        meta_file_path = INCOMING_FOLDER / meta_filename
        
        if not meta_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Metadata file '{meta_filename}' not found"
            )
        
        # Validate before processing
        validation = invoice_processor.validate_invoice_pair(meta_file_path)
        
        if not validation.get('valid', False):
            return {
                "success": False,
                "message": "Invoice validation failed",
                "validation": validation
            }
        
        # Process the invoice pair
        result = invoice_processor.process_invoice_pair(meta_file_path)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-all")
async def process_all_invoices():
    """
    Process all invoice pairs in the incoming folder
    Only processes invoices that have both metadata and attachment files
    """
    try:
        result = invoice_processor.process_all_invoices()
        return result
    except Exception as e:
        logger.error(f"Error processing all invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_invoice_stats():
    """Get statistics about invoices in the system"""
    try:
        meta_files = invoice_processor.find_meta_files()
        pending_invoices = invoice_processor.get_pending_invoices()
        
        valid_invoices = [inv for inv in pending_invoices if inv.get('valid', False)]
        invalid_invoices = [inv for inv in pending_invoices if not inv.get('valid', False)]
        
        # Count by language
        languages = {}
        for inv in valid_invoices:
            lang = inv.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "pending": {
                "total": len(meta_files),
                "valid": len(valid_invoices),
                "invalid": len(invalid_invoices)
            },
            "by_language": languages,
            "invalid_reasons": [
                {
                    "meta_file": inv.get('meta_file'),
                    "missing_files": inv.get('missing_files', [])
                }
                for inv in invalid_invoices if inv.get('missing_files')
            ]
        }
    except Exception as e:
        logger.error(f"Error getting invoice stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))