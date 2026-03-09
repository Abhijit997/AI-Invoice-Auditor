"""
Invoice File Processor - Handles paired processing of .meta.json and invoice files
"""
from pathlib import Path
import json
import shutil
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from agenticaicapstone.src.rag.vector_db_loader import VectorDBLoader

logger = logging.getLogger(__name__)

# Allowed file extensions for invoice attachments
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.json', '.csv', '.jpg', '.png'}


class InvoiceMetadata:
    """Represents invoice metadata from .meta.json files"""
    
    def __init__(self, meta_file_path: Path):
        self.meta_file_path = meta_file_path
        self.data = self._load_metadata()
        
    def _load_metadata(self) -> dict:
        """Load and parse the .meta.json file"""
        try:
            with open(self.meta_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {self.meta_file_path}: {e}")
            raise
    
    @property
    def sender(self) -> str:
        return self.data.get('sender', 'unknown')
    
    @property
    def subject(self) -> str:
        return self.data.get('subject', '')
    
    @property
    def received_timestamp(self) -> str:
        return self.data.get('received_timestamp', '')
    
    @property
    def language(self) -> str:
        return self.data.get('language', 'unknown')
    
    @property
    def attachments(self) -> List[str]:
        """Get list of attachment filenames"""
        attachments = self.data.get('attachments', [])
        return attachments if isinstance(attachments, list) else [attachments]
    
    def get_attachment_paths(self, base_folder: Path) -> List[Path]:
        """Get full paths to attachment files"""
        paths = []
        for attachment in self.attachments:
            path = base_folder / attachment
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"Attachment file not found: {path}")
        return paths


class InvoiceProcessor:
    """Processes invoice files with their metadata"""
    
    def __init__(self, incoming_folder: Path, processed_folder: Path, unprocessed_folder: Path = None):
        self.incoming_folder = incoming_folder
        self.processed_folder = processed_folder
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        
        # Unprocessed folder for files that fail vector DB loading
        self.unprocessed_folder = unprocessed_folder or (incoming_folder.parent / "unprocessed")
        self.unprocessed_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector DB loader
        self.vector_db_loader = VectorDBLoader()
    
    def find_meta_files(self) -> List[Path]:
        """Find all .meta.json files in the incoming folder"""
        return list(self.incoming_folder.glob("*.meta.json"))
    
    def get_invoice_pair(self, meta_file: Path) -> Tuple[InvoiceMetadata, List[Path]]:
        """
        Get metadata and corresponding attachment files for an invoice
        
        Returns:
            Tuple of (metadata, list of attachment paths)
        """
        metadata = InvoiceMetadata(meta_file)
        attachment_paths = metadata.get_attachment_paths(self.incoming_folder)
        return metadata, attachment_paths
    
    def validate_invoice_pair(self, meta_file: Path) -> Dict[str, any]:
        """
        Validate that invoice has all required files and correct file types
        
        Returns:
            Dict with validation status and details
        """
        try:
            metadata = InvoiceMetadata(meta_file)
            attachment_paths = metadata.get_attachment_paths(self.incoming_folder)
            
            missing_files = []
            invalid_extensions = []
            
            for attachment_name in metadata.attachments:
                attachment_path = self.incoming_folder / attachment_name
                
                # Check if file exists
                if not attachment_path.exists():
                    missing_files.append(attachment_name)
                else:
                    # Check if file extension is allowed
                    file_ext = attachment_path.suffix.lower()
                    if file_ext not in ALLOWED_EXTENSIONS:
                        invalid_extensions.append({
                            "file": attachment_name,
                            "extension": file_ext
                        })
            
            errors = []
            if missing_files:
                errors.append(f"Missing files: {', '.join(missing_files)}")
            if invalid_extensions:
                invalid_list = [f"{item['file']} ({item['extension']})" for item in invalid_extensions]
                errors.append(f"Invalid file types: {', '.join(invalid_list)}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
            
            return {
                "meta_file": meta_file.name,
                "valid": len(missing_files) == 0 and len(invalid_extensions) == 0,
                "sender": metadata.sender,
                "language": metadata.language,
                "expected_attachments": metadata.attachments,
                "found_attachments": [p.name for p in attachment_paths],
                "missing_files": missing_files,
                "invalid_extensions": invalid_extensions,
                "errors": errors,
                "total_files": len(attachment_paths) + 1  # +1 for meta file
            }
        except Exception as e:
            logger.error(f"Validation failed for {meta_file}: {e}")
            return {
                "meta_file": meta_file.name,
                "valid": False,
                "error": str(e)
            }
    
    def load_invoice_to_vector_db(self, meta_file: Path, attachment_paths: List[Path], metadata: 'InvoiceMetadata') -> str:
        """
        Extract contents from invoice files and load into vector database
        
        Collection locks prevent race conditions during parallel processing.
        
        Args:
            meta_file: Path to the metadata file
            attachment_paths: List of paths to attachment files
            metadata: InvoiceMetadata object with parsed metadata
            
        Returns:
            str: "OK" if successful, error message if failed
        """
        # Convert metadata to dictionary (exclude attachments - not part of schema)
        metadata_dict = {
            "sender": metadata.sender,
            "subject": metadata.subject,
            "language": metadata.language,
            "received_timestamp": metadata.received_timestamp
        }
        
        # Call VectorDBLoader to handle the 3 steps:
        # 1. Extract content from files
        # 2. Create embeddings
        # 3. Store in vector database
        return self.vector_db_loader.load_invoice(meta_file, attachment_paths, metadata_dict)
    
    def process_invoice_pair(self, meta_file: Path) -> Dict[str, any]:
        """
        Process an invoice pair (metadata + attachments)
        Move both meta file and attachment(s) to processed folder
        
        Collection locks prevent race conditions during parallel processing.
        
        Args:
            meta_file: Path to metadata file
        
        Returns:
            Dict with processing result
        """
        result = {
            "meta_file": meta_file.name,
            "success": False,
            "moved_files": [],
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Load metadata
            metadata = InvoiceMetadata(meta_file)
            result["sender"] = metadata.sender
            result["language"] = metadata.language
            result["attachments"] = metadata.attachments
            
            # Get attachment files
            attachment_paths = metadata.get_attachment_paths(self.incoming_folder)
            
            # Check if all attachments exist and have valid extensions
            missing = []
            invalid_types = []
            
            for attachment_name in metadata.attachments:
                attachment_path = self.incoming_folder / attachment_name
                
                if not attachment_path.exists():
                    missing.append(attachment_name)
                else:
                    # Check file extension
                    file_ext = attachment_path.suffix.lower()
                    if file_ext not in ALLOWED_EXTENSIONS:
                        invalid_types.append(f"{attachment_name} ({file_ext})")
            
            if missing:
                result["errors"].append(f"Missing attachment files: {', '.join(missing)}")
                result["missing_files"] = missing
                return result
            
            if invalid_types:
                result["errors"].append(f"Invalid file types: {', '.join(invalid_types)}. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
                result["invalid_file_types"] = invalid_types
                return result
            
            # Load invoice into vector database
            vector_db_result = self.load_invoice_to_vector_db(meta_file, attachment_paths, metadata)
            
            # Determine destination folder based on vector DB loading result
            if vector_db_result == "OK":
                destination_folder = self.processed_folder
                result["vector_db_loaded"] = True
            else:
                destination_folder = self.unprocessed_folder
                result["vector_db_loaded"] = False
                result["vector_db_error"] = vector_db_result
                result["errors"].append(f"Vector DB loading failed: {vector_db_result}")
                logger.error(f"Failed to load {meta_file.name} to vector database: {vector_db_result}")
            
            # Move metadata file
            dest_meta = self._get_unique_destination(meta_file.name, destination_folder)
            shutil.move(str(meta_file), str(dest_meta))
            result["moved_files"].append({
                "source": meta_file.name,
                "destination": dest_meta.name,
                "destination_folder": "processed" if vector_db_result == "OK" else "unprocessed",
                "type": "metadata"
            })
            
            # Move attachment files
            for attachment_path in attachment_paths:
                dest_attachment = self._get_unique_destination(attachment_path.name, destination_folder)
                shutil.move(str(attachment_path), str(dest_attachment))
                result["moved_files"].append({
                    "source": attachment_path.name,
                    "destination": dest_attachment.name,
                    "destination_folder": "processed" if vector_db_result == "OK" else "unprocessed",
                    "type": "attachment"
                })
            
            # Mark as success only if vector DB loading succeeded
            result["success"] = (vector_db_result == "OK")
            result["total_files_moved"] = len(result["moved_files"])
            
            if vector_db_result != "OK":
                result["message"] = "Files moved to unprocessed folder due to vector DB loading failure"
            
        except Exception as e:
            error_msg = f"Failed to process {meta_file.name}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
        
        return result
    
    def process_all_invoices(self, max_workers: int = 8, api_base_url: str = "http://localhost:8000") -> Dict[str, any]:
        """
        Process all invoice pairs in the incoming folder using parallel processing
        
        Collection locking prevents race conditions:
        - Workers generate embeddings in parallel (slow part)
        - Collection locks serialize ID generation and inserts (fast part)
        - No wasted IDs from over-estimation
        
        Args:
            max_workers: Maximum number of parallel threads (default: 8)
            api_base_url: Base URL for API endpoints (default: http://localhost:8000)
        
        Returns:
            Summary of processing results
        """
        meta_files = self.find_meta_files()
        
        if not meta_files:
            return {
                "success": True,
                "message": "No invoice files to process",
                "processed_count": 0,
                "failed_count": 0,
                "processed_invoices": [],
                "failed_invoices": []
            }
        
        logger.info(f"Processing {len(meta_files)} invoices in parallel with {max_workers} workers (using collection locks)")
        
        processed_invoices = []
        failed_invoices = []
        
        # Process invoices in parallel using ThreadPoolExecutor
        # Collection locks in VectorDBLoader prevent race conditions
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks - locks handle race conditions
            future_to_meta = {}
            for meta_file in meta_files:
                future = executor.submit(self.process_invoice_pair, meta_file)
                future_to_meta[future] = meta_file
            
            # Collect results as they complete
            for future in as_completed(future_to_meta):
                meta_file = future_to_meta[future]
                try:
                    result = future.result()
                    
                    if result["success"]:
                        processed_invoices.append(result)
                    else:
                        failed_invoices.append(result)
                        
                except Exception as e:
                    error_msg = f"Exception processing {meta_file.name}: {str(e)}"
                    logger.error(error_msg)
                    failed_invoices.append({
                        "meta_file": meta_file.name,
                        "success": False,
                        "errors": [error_msg],
                        "timestamp": datetime.now().isoformat()
                    })
        
        return {
            "success": len(failed_invoices) == 0,
            "message": f"Processed {len(processed_invoices)} invoices, {len(failed_invoices)} failed",
            "processed_count": len(processed_invoices),
            "failed_count": len(failed_invoices),
            "processed_invoices": processed_invoices,
            "failed_invoices": failed_invoices,
            "timestamp": datetime.now().isoformat(),
            "max_workers": max_workers
        }
    
    def get_pending_invoices(self) -> List[Dict[str, any]]:
        """
        Get list of all pending invoices with validation status
        
        Returns:
            List of invoice information
        """
        meta_files = self.find_meta_files()
        invoices = []
        
        for meta_file in meta_files:
            validation = self.validate_invoice_pair(meta_file)
            invoices.append(validation)
        
        return invoices
    
    def _get_unique_destination(self, filename: str, destination_folder: Path = None) -> Path:
        """
        Get a unique destination path, adding timestamp if file exists
        
        Args:
            filename: Name of the file
            destination_folder: Target folder (defaults to processed_folder)
        """
        if destination_folder is None:
            destination_folder = self.processed_folder
        
        dest_path = destination_folder / filename
        
        if dest_path.exists():
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = filename.rsplit('.', 1)
            if len(name_parts) == 2:
                new_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            else:
                new_filename = f"{filename}_{timestamp}"
            dest_path = self.processed_folder / new_filename
        
        return dest_path