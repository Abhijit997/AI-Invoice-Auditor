"""
Chroma Vector Store for Invoice RAG System

Manages six collections:
1. invoices - Invoice documents with extracted data
2. vendors - Vendor master data
3. skus - SKU/Item master data
4. invoices_stage - Staging area for invoices before final processing
5. vendors_stage - Staging area for vendors before final processing
6. skus_stage - Staging area for SKUs before final processing
"""

import chromadb
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agenticaicapstone.src.utils.azure_openai_client import AzureOpenAIClient

logger = logging.getLogger(__name__)


class InvoiceVectorStore:
    """
    Vector store for invoice processing system using ChromaDB
    
    Storage location: data/vector_store/
    Collections:
        - invoices: Invoice documents with extracted data
        - vendors: Vendor master data from ERP
        - skus: SKU/Item master data from ERP
        - invoices_stage: Staging area for invoices before final processing
        - vendors_stage: Staging area for vendors before final processing
        - skus_stage: Staging area for SKUs before final processing
    """
    
    def __init__(self, persist_directory: str = "data/vector_store"):
        """
        Initialize Chroma vector store with persistent storage
        
        Args:
            persist_directory: Directory path for storing vector data
                              Default: data/vector_store/
        """
        # Convert to absolute path and create directory
        self.persist_dir = Path(persist_directory).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence and telemetry disabled
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
        # Initialize Azure OpenAI client for embeddings
        try:
            self.azure_client = AzureOpenAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
            self.azure_client = None
        
        # Create/get collections
        self.invoices_collection = self._init_collection(
            name="invoices",
            metadata={"description": "Invoice documents with extracted data"}
        )
        
        self.vendors_collection = self._init_collection(
            name="vendors",
            metadata={"description": "Vendor master data from ERP"}
        )
        
        self.skus_collection = self._init_collection(
            name="skus",
            metadata={"description": "SKU/Item master data from ERP"}
        )
        
        # Create/get staging collections
        self.invoices_stage_collection = self._init_collection(
            name="invoices_stage",
            metadata={"description": "Staging area for invoices before final processing"}
        )
        
        self.vendors_stage_collection = self._init_collection(
            name="vendors_stage",
            metadata={"description": "Staging area for vendors before final processing"}
        )
        
        self.skus_stage_collection = self._init_collection(
            name="skus_stage",
            metadata={"description": "Staging area for SKUs before final processing"}
        )
    
    def _init_collection(self, name: str, metadata: Dict) -> chromadb.Collection:
        """
        Initialize or get existing collection
        
        Args:
            name: Collection name
            metadata: Collection metadata
            
        Returns:
            ChromaDB collection object
        """
        return self.client.get_or_create_collection(
            name=name,
            metadata={
                **metadata,
                "hnsw:space": "cosine"  # Use cosine similarity for search
            }
        )
    
    # ==================== INVOICE OPERATIONS ====================
    
    def add_invoice(
        self,
        invoice_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        Add invoice document to vector store
        
        Args:
            invoice_id: Unique invoice identifier (e.g., "INV-001")
            content: Invoice text content (for embedding)
            metadata: Invoice metadata dict with fields:
                - invoice_number: str
                - vendor_name: str
                - vendor_id: str (optional)
                - amount: float
                - currency: str
                - date: str
                - po_number: str (optional)
                - language: str (optional)
            embedding: Pre-computed embedding vector (optional)
                      If not provided, ChromaDB will generate embeddings
        """
        try:
            add_params = {
                "documents": [content],
                "metadatas": [metadata],
                "ids": [invoice_id]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.invoices_collection.add(**add_params)
            logger.info(f"Added invoice: {invoice_id}")
        except Exception as e:
            logger.error(f"Failed to add invoice {invoice_id}: {e}")
            raise
    
    def search_invoices(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """
        Search invoices by semantic similarity
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_dict: Metadata filters (e.g., {"vendor_name": "Acme"})
            query_embedding: Pre-computed query embedding (optional)
                           If not provided, will generate using Azure OpenAI
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        # Use provided embedding or generate with Azure
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
                logger.info("Generated query embedding with Azure OpenAI")
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.invoices_collection.query(**query_params)
    
    def get_invoice(self, invoice_id: str) -> Dict:
        """Get specific invoice by ID"""
        return self.invoices_collection.get(ids=[invoice_id])
    
    def update_invoice(
        self,
        invoice_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update invoice content, metadata, or embedding"""
        update_dict = {"ids": [invoice_id]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.invoices_collection.update(**update_dict)
        logger.info(f"Updated invoice: {invoice_id}")
    
    def delete_invoice(self, invoice_id: str) -> None:
        """Delete invoice from vector store"""
        self.invoices_collection.delete(ids=[invoice_id])
        logger.info(f"Deleted invoice: {invoice_id}")
    
    # ==================== VENDOR OPERATIONS ====================
    
    def add_vendor(
        self,
        vendor_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        Add vendor to vector store
        
        Args:
            vendor_id: Unique vendor identifier (e.g., "VEND-001")
            content: Vendor description text (for embedding)
            metadata: Vendor metadata dict with fields:
                - vendor_name: str
                - country: str
                - currency: str
                - full_address: str
                - src_invoice_id: str (foreign key to source invoice)
                - contact_info: str (optional)
            embedding: Pre-computed embedding vector (optional)
                      If not provided, ChromaDB will generate embeddings
        """
        try:
            add_params = {
                "documents": [content],
                "metadatas": [metadata],
                "ids": [vendor_id]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.vendors_collection.add(**add_params)
            logger.info(f"Added vendor: {vendor_id}")
        except Exception as e:
            logger.error(f"Failed to add vendor {vendor_id}: {e}")
            raise
    
    def search_vendors(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """
        Search vendors by semantic similarity
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_dict: Metadata filters (e.g., {"country": "USA"})
            query_embedding: Pre-computed query embedding (optional)
                           If not provided, will generate using Azure OpenAI
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        # Use provided embedding or generate with Azure
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
                logger.info("Generated query embedding with Azure OpenAI")
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.vendors_collection.query(**query_params)
    
    def get_vendor(self, vendor_id: str) -> Dict:
        """Get specific vendor by ID"""
        return self.vendors_collection.get(ids=[vendor_id])
    
    def update_vendor(
        self,
        vendor_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update vendor content, metadata, or embedding"""
        update_dict = {"ids": [vendor_id]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.vendors_collection.update(**update_dict)
        logger.info(f"Updated vendor: {vendor_id}")
    
    def delete_vendor(self, vendor_id: str) -> None:
        """Delete vendor from vector store"""
        self.vendors_collection.delete(ids=[vendor_id])
        logger.info(f"Deleted vendor: {vendor_id}")
    
    # ==================== SKU OPERATIONS ====================
    
    def add_sku(
        self,
        item_code: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        Add SKU/item to vector store
        
        Args:
            item_code: Unique item code (e.g., "SKU-001")
            content: Item description text (for embedding)
            metadata: SKU metadata dict with fields:
                - description: str
                - category: str
                - uom: str
                - gst_rate: float
                - src_invoice_id: str (foreign key to source invoice)
                - unit_price: float (optional)
            embedding: Pre-computed embedding vector (optional)
                      If not provided, ChromaDB will generate embeddings
        """
        try:
            add_params = {
                "documents": [content],
                "metadatas": [metadata],
                "ids": [item_code]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.skus_collection.add(**add_params)
            logger.info(f"Added SKU: {item_code}")
        except Exception as e:
            logger.error(f"Failed to add SKU {item_code}: {e}")
            raise
    
    def search_skus(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """
        Search SKUs by semantic similarity
        
        Args:
            query: Search query text (e.g., "safety helmets")
            n_results: Number of results to return
            filter_dict: Metadata filters (e.g., {"category": "Safety"})
            query_embedding: Pre-computed query embedding (optional)
                           If not provided, will generate using Azure OpenAI
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        # Use provided embedding or generate with Azure
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
                logger.info("Generated query embedding with Azure OpenAI")
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.skus_collection.query(**query_params)
    
    def get_sku(self, item_code: str) -> Dict:
        """Get specific SKU by item code"""
        return self.skus_collection.get(ids=[item_code])
    
    def update_sku(
        self,
        item_code: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update SKU content, metadata, or embedding"""
        update_dict = {"ids": [item_code]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.skus_collection.update(**update_dict)
        logger.info(f"Updated SKU: {item_code}")
    
    def delete_sku(self, item_code: str) -> None:
        """Delete SKU from vector store"""
        self.skus_collection.delete(ids=[item_code])
        logger.info(f"Deleted SKU: {item_code}")
    
    # ==================== INVOICE STAGE OPERATIONS ====================
    
    def add_invoice_stage(
        self,
        invoice_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add invoice to staging collection"""
        try:
            # Convert None values to empty strings (ChromaDB doesn't accept None)
            sanitized_metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
            
            add_params = {
                "documents": [content],
                "metadatas": [sanitized_metadata],
                "ids": [invoice_id]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.invoices_stage_collection.add(**add_params)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add invoice to stage {invoice_id}: {e}")
            return False
    
    def search_invoices_stage(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """Search staged invoices by semantic similarity"""
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.invoices_stage_collection.query(**query_params)
    
    def get_invoice_stage(self, invoice_id: str) -> Dict:
        """Get specific staged invoice by ID"""
        return self.invoices_stage_collection.get(ids=[invoice_id])
    
    def update_invoice_stage(
        self,
        invoice_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update staged invoice content, metadata, or embedding"""
        update_dict = {"ids": [invoice_id]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.invoices_stage_collection.update(**update_dict)
        return True
    
    def delete_invoice_stage(self, invoice_id: str) -> None:
        """Delete staged invoice from vector store"""
        self.invoices_stage_collection.delete(ids=[invoice_id])
        return True
    
    # ==================== VENDOR STAGE OPERATIONS ====================
    
    def add_vendor_stage(
        self,
        vendor_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add vendor to staging collection"""
        try:
            # Convert None values to empty strings (ChromaDB doesn't accept None)
            sanitized_metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
            
            add_params = {
                "documents": [content],
                "metadatas": [sanitized_metadata],
                "ids": [vendor_id]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.vendors_stage_collection.add(**add_params)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add vendor to stage {vendor_id}: {e}")
            return False
    
    def search_vendors_stage(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """Search staged vendors by semantic similarity"""
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.vendors_stage_collection.query(**query_params)
    
    def get_vendor_stage(self, vendor_id: str) -> Dict:
        """Get specific staged vendor by ID"""
        return self.vendors_stage_collection.get(ids=[vendor_id])
    
    def update_vendor_stage(
        self,
        vendor_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update staged vendor content, metadata, or embedding"""
        update_dict = {"ids": [vendor_id]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.vendors_stage_collection.update(**update_dict)
        return True
    
    def delete_vendor_stage(self, vendor_id: str) -> None:
        """Delete staged vendor from vector store"""
        self.vendors_stage_collection.delete(ids=[vendor_id])
        return True
    
    # ==================== SKU STAGE OPERATIONS ====================
    
    def add_sku_stage(
        self,
        item_code: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add SKU to staging collection"""
        try:
            # Convert None values to empty strings (ChromaDB doesn't accept None)
            sanitized_metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
            
            add_params = {
                "documents": [content],
                "metadatas": [sanitized_metadata],
                "ids": [item_code]
            }
            
            if embedding is not None:
                add_params["embeddings"] = [embedding]
            
            self.skus_stage_collection.add(**add_params)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add SKU to stage {item_code}: {e}")
            return False
    
    def search_skus_stage(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict:
        """Search staged SKUs by semantic similarity"""
        query_params = {
            "n_results": n_results,
            "where": filter_dict
        }
        
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        elif self.azure_client:
            try:
                embedding = self.azure_client.create_embeddings(query)
                query_params["query_embeddings"] = [embedding]
            except Exception as e:
                logger.warning(f"Failed to generate embedding with Azure: {e}. Falling back to ChromaDB default.")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        return self.skus_stage_collection.query(**query_params)
    
    def get_sku_stage(self, item_code: str) -> Dict:
        """Get specific staged SKU by item code"""
        return self.skus_stage_collection.get(ids=[item_code])
    
    def update_sku_stage(
        self,
        item_code: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update staged SKU content, metadata, or embedding"""
        update_dict = {"ids": [item_code]}
        if content:
            update_dict["documents"] = [content]
        if metadata:
            update_dict["metadatas"] = [metadata]
        if embedding is not None:
            update_dict["embeddings"] = [embedding]
        
        self.skus_stage_collection.update(**update_dict)
        return True
    
    def delete_sku_stage(self, item_code: str) -> None:
        """Delete staged SKU from vector store"""
        self.skus_stage_collection.delete(ids=[item_code])
        return True
    
    # ==================== UTILITY METHODS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dict with collection counts and storage info
        """
        return {
            "persist_directory": str(self.persist_dir),
            "collections": {
                "invoices": self.invoices_collection.count(),
                "vendors": self.vendors_collection.count(),
                "skus": self.skus_collection.count(),
                "invoices_stage": self.invoices_stage_collection.count(),
                "vendors_stage": self.vendors_stage_collection.count(),
                "skus_stage": self.skus_stage_collection.count()
            },
            "total_documents": (
                self.invoices_collection.count() +
                self.vendors_collection.count() +
                self.skus_collection.count() +
                self.invoices_stage_collection.count() +
                self.vendors_stage_collection.count() +
                self.skus_stage_collection.count()
            )
        }
    
    def list_collections(self) -> List[str]:
        """List all collection names"""
        return [coll.name for coll in self.client.list_collections()]
    
    def clear_collection_data(self, collection_name: str) -> int:
        """
        Clear all data from a collection without deleting/recreating it
        
        Args:
            collection_name: Name of collection to clear
            
        Returns:
            Number of documents deleted
        """
        allowed_collections = ["invoices", "vendors", "skus", "invoices_stage", "vendors_stage", "skus_stage"]
        if collection_name not in allowed_collections:
            raise ValueError(f"Invalid collection name: {collection_name}. Allowed: {allowed_collections}")
        
        # Get the collection
        collection = getattr(self, f"{collection_name}_collection")
        
        # Get all IDs
        all_data = collection.get()
        ids = all_data.get('ids', [])
        count = len(ids)
        
        # Delete all documents
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Cleared {count} documents from {collection_name}")
        
        return count
    
    def clear_all_staging_data(self) -> Dict[str, int]:
        """
        Clear all data from staging collections only
        
        Returns:
            Dict with collection names and count of documents deleted
        """
        staging_collections = ["invoices_stage", "vendors_stage", "skus_stage"]
        results = {}
        
        for collection_name in staging_collections:
            count = self.clear_collection_data(collection_name)
            results[collection_name] = count
        
        logger.info(f"Cleared all staging collections. Total deleted: {sum(results.values())}")
        return results
    
    def clear_all_production_data(self) -> Dict[str, int]:
        """
        Clear all data from production collections only
        
        Returns:
            Dict with collection names and count of documents deleted
        """
        production_collections = ["invoices", "vendors", "skus"]
        results = {}
        
        for collection_name in production_collections:
            count = self.clear_collection_data(collection_name)
            results[collection_name] = count
        
        logger.info(f"Cleared all production collections. Total deleted: {sum(results.values())}")
        return results
    
    def reset_all(self) -> None:
        """Reset entire database (CAUTION: deletes all data!)"""
        self.client.reset()
        logger.warning("Reset all collections")
        
        # Reinitialize
        self.__init__(str(self.persist_dir))
    
    def wipe_storage(self) -> None:
        """
        Completely wipe the persist directory and reinitialize (CAUTION: deletes all files!)
        This removes all ChromaDB storage files from disk, not just the data.
        
        NOTE: This must be called when NO servers are running (stop uvicorn first)
        """
        import shutil
        import time
        import gc
        import os
        
        persist_path = self.persist_dir
        logger.warning(f"Wiping entire storage directory: {persist_path}")
        
        # Step 1: Close all ChromaDB connections to release file locks
        try:
            # Clear all collection references
            self.invoices_collection = None
            self.vendors_collection = None
            self.skus_collection = None
            self.invoices_stage_collection = None
            self.vendors_stage_collection = None
            self.skus_stage_collection = None
            
            # Clear Azure client reference
            self.azure_client = None
            
            # Delete the ChromaDB client to close SQLite connection
            if hasattr(self, 'client') and self.client is not None:
                del self.client
            self.client = None
            
            # Force garbage collection to ensure all file handles are released
            gc.collect()
            
            # Longer delay for Windows to release the file handle
            time.sleep(1.0)
            
            logger.info("Closed all ChromaDB connections")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")
        
        # Step 2: Try to remove directory, with retry logic for Windows file locks
        if persist_path.exists():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # On Windows, use onerror callback to handle read-only files
                    def handle_remove_readonly(func, path, exc):
                        """Error handler for Windows readonly files"""
                        if os.path.exists(path):
                            os.chmod(path, 0o777)
                            func(path)
                    
                    shutil.rmtree(persist_path, onerror=handle_remove_readonly)
                    logger.warning(f"Deleted directory: {persist_path}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries}: File still locked, waiting...")
                        time.sleep(2.0)
                        gc.collect()
                    else:
                        logger.error(f"Failed to delete directory after {max_retries} attempts: {e}")
                        logger.error("Make sure all servers (uvicorn, streamlit) are stopped!")
                        raise
                except Exception as e:
                    logger.error(f"Failed to delete directory: {e}")
                    raise
        
        # Step 3: Recreate directory and reinitialize everything
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Reinitialize the client and collections
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
        # Recreate all collections
        self.invoices_collection = self._init_collection(
            name="invoices",
            metadata={"description": "Invoice documents with extracted data"}
        )
        self.vendors_collection = self._init_collection(
            name="vendors",
            metadata={"description": "Vendor master data from ERP"}
        )
        self.skus_collection = self._init_collection(
            name="skus",
            metadata={"description": "SKU/Item master data from ERP"}
        )
        self.invoices_stage_collection = self._init_collection(
            name="invoices_stage",
            metadata={"description": "Staging area for invoices before final processing"}
        )
        self.vendors_stage_collection = self._init_collection(
            name="vendors_stage",
            metadata={"description": "Staging area for vendors before final processing"}
        )
        self.skus_stage_collection = self._init_collection(
            name="skus_stage",
            metadata={"description": "Staging area for SKUs before final processing"}
        )
        
        logger.info("Storage wiped and reinitialized with empty collections")


# Convenience function for quick initialization
def get_vector_store(persist_directory: str = "data/vector_store") -> InvoiceVectorStore:
    """
    Get or create vector store instance
    
    Args:
        persist_directory: Storage location for vector data
        
    Returns:
        InvoiceVectorStore instance
    """
    return InvoiceVectorStore(persist_directory)