"""FastAPI application for Invoice File Management and Vector Store"""
import os
import warnings

# Disable SSL warnings for corporate environments
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Disable SSL verification at urllib3 level
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Monkey-patch requests to disable SSL verification for LangSmith
import requests
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Load environment variables BEFORE any other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pathlib import Path
import logging
import sys
from contextlib import asynccontextmanager

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routers import invoices_router, vector_router, review_router
from api.processor import InvoiceProcessor
from api.file_watcher import FileWatcherService

# Configure logging with console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

# Log startup message
logger.info("=" * 60)
logger.info("AI Invoice Auditor API Starting...")
logger.info("Logging configured successfully")
logger.info("=" * 60)

# Global file watcher instance
file_watcher = FileWatcherService(incoming_folder="data/incoming", debounce_seconds=5)


# Lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("=" * 60)
    logger.info("FastAPI application started successfully")
    logger.info("API is ready to handle requests")
    logger.info("=" * 60)
    
    # Start file watcher for automatic processing
    try:
        processor = InvoiceProcessor(
            incoming_folder=Path("data/incoming"),
            processed_folder=Path("data/processed"),
            unprocessed_folder=Path("data/unprocessed")
        )
        file_watcher.start(processor)
    except Exception as e:
        logger.error(f"Failed to start file watcher: {e}")
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Stopping file watcher...")
    file_watcher.stop()
    logger.info("FastAPI application shutting down...")
    logger.info("=" * 60)


app = FastAPI(
    title="AI Invoice Auditor API",
    description="API for managing invoice files, ERP data, and vector store operations",
    version="2.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(invoices_router)
app.include_router(vector_router)
app.include_router(review_router)


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "status": "healthy",
        "service": "AI Invoice Auditor API",
        "version": "2.0.0",
        "endpoints": {
            "invoices": [
                "GET /invoices/pending - Get pending invoices",
                "POST /invoices/process/{meta_filename} - Process single invoice",
                "POST /invoices/process-all - Process all invoices",
                "GET /invoices/stats - Get invoice statistics"
            ],
            "vector_store": {
                "invoices": [
                    "POST /vector/invoices - Add invoice to vector store",
                    "POST /vector/invoices/search - Search invoices",
                    "GET /vector/invoices/{invoice_id} - Get invoice by ID",
                    "PUT /vector/invoices/{invoice_id} - Update invoice",
                    "DELETE /vector/invoices/{invoice_id} - Delete invoice"
                ],
                "vendors": [
                    "POST /vector/vendors - Add vendor",
                    "POST /vector/vendors/search - Search vendors",
                    "GET /vector/vendors/{vendor_id} - Get vendor by ID",
                    "PUT /vector/vendors/{vendor_id} - Update vendor",
                    "DELETE /vector/vendors/{vendor_id} - Delete vendor"
                ],
                "skus": [
                    "POST /vector/skus - Add SKU",
                    "POST /vector/skus/search - Search SKUs",
                    "GET /vector/skus/{item_code} - Get SKU by code",
                    "PUT /vector/skus/{item_code} - Update SKU",
                    "DELETE /vector/skus/{item_code} - Delete SKU"
                ],
                "invoices_stage": [
                    "POST /vector/invoices-stage - Add invoice to staging",
                    "POST /vector/invoices-stage/search - Search staged invoices",
                    "GET /vector/invoices-stage/{invoice_id} - Get staged invoice by ID",
                    "PUT /vector/invoices-stage/{invoice_id} - Update staged invoice",
                    "DELETE /vector/invoices-stage/{invoice_id} - Delete staged invoice"
                ],
                "vendors_stage": [
                    "POST /vector/vendors-stage - Add vendor to staging",
                    "POST /vector/vendors-stage/search - Search staged vendors",
                    "GET /vector/vendors-stage/{vendor_id} - Get staged vendor by ID",
                    "PUT /vector/vendors-stage/{vendor_id} - Update staged vendor",
                    "DELETE /vector/vendors-stage/{vendor_id} - Delete staged vendor"
                ],
                "skus_stage": [
                    "POST /vector/skus-stage - Add SKU to staging",
                    "POST /vector/skus-stage/search - Search staged SKUs",
                    "GET /vector/skus-stage/{item_code} - Get staged SKU by code",
                    "PUT /vector/skus-stage/{item_code} - Update staged SKU",
                    "DELETE /vector/skus-stage/{item_code} - Delete staged SKU"
                ],
                "utility": [
                    "GET /vector/stats - Get vector store statistics",
                    "GET /vector/collections - List all collections",
                    "DELETE /vector/collections/reset - Reset collection(s) (deletes data, keeps files)",
                    "DELETE /vector/storage/wipe - Wipe entire storage (deletes all files)"
                ]
            },
            "file_watcher": [
                "GET /file-watcher/status - Check file watcher status",
                "POST /file-watcher/start - Start file watcher",
                "POST /file-watcher/stop - Stop file watcher"
            ],
            "review": [
                "GET /review/pending - List pending invoices for review",
                "GET /review/invoice/{invoice_id} - Get full invoice+vendors+SKUs",
                "POST /review/invoice/{invoice_id}/approve - Approve & promote to production",
                "POST /review/invoice/{invoice_id}/reject - Reject invoice",
                "GET /review/file/{filename} - Get file info for preview"
            ]
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/file-watcher/status")
async def get_file_watcher_status():
    """Get the current status of the file watcher"""
    return {
        "running": file_watcher.is_running(),
        "incoming_folder": str(file_watcher.incoming_folder),
        "debounce_seconds": file_watcher.debounce_seconds,
        "description": "Automatically processes new invoices when files are added to incoming folder"
    }


@app.post("/file-watcher/start")
async def start_file_watcher():
    """Start the file watcher for automatic processing"""
    try:
        if file_watcher.is_running():
            return {
                "success": False,
                "message": "File watcher is already running"
            }
        
        processor = InvoiceProcessor(
            incoming_folder=Path("data/incoming"),
            processed_folder=Path("data/processed"),
            unprocessed_folder=Path("data/unprocessed")
        )
        file_watcher.start(processor)
        
        return {
            "success": True,
            "message": "File watcher started successfully",
            "incoming_folder": str(file_watcher.incoming_folder)
        }
    except Exception as e:
        logger.error(f"Failed to start file watcher: {e}")
        return {
            "success": False,
            "message": f"Failed to start file watcher: {str(e)}"
        }


@app.post("/file-watcher/stop")
async def stop_file_watcher():
    """Stop the file watcher"""
    try:
        if not file_watcher.is_running():
            return {
                "success": False,
                "message": "File watcher is not running"
            }
        
        file_watcher.stop()
        
        return {
            "success": True,
            "message": "File watcher stopped successfully"
        }
    except Exception as e:
        logger.error(f"Failed to stop file watcher: {e}")
        return {
            "success": False,
            "message": f"Failed to stop file watcher: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["data/**"]
    )