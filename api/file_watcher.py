"""
File Watcher for Automatic Invoice Processing

Monitors the data/incoming folder for new invoice files and automatically
triggers processing when new .meta.json files are detected.
"""
import logging
import time
from pathlib import Path
from threading import Thread, Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class InvoiceFileHandler(FileSystemEventHandler):
    """Handler for file system events in the incoming folder"""
    
    def __init__(self, processor, debounce_seconds: int = 5):
        """
        Initialize the file handler
        
        Args:
            processor: InvoiceProcessor instance
            debounce_seconds: Time to wait before processing after file changes
        """
        super().__init__()
        self.processor = processor
        self.debounce_seconds = debounce_seconds
        self.pending_process = False
        self.last_event_time = 0
        self.lock = Lock()
        self.process_thread = None
        
    def on_created(self, event):
        """Called when a file or directory is created"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only trigger on .meta.json files
        if file_path.suffix == '.json' and '.meta.' in file_path.name:
            logger.info(f"New invoice file detected: {file_path.name}")
            self._schedule_processing()
    
    def on_modified(self, event):
        """Called when a file or directory is modified"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only trigger on .meta.json files
        if file_path.suffix == '.json' and '.meta.' in file_path.name:
            logger.info(f"Invoice file modified: {file_path.name}")
            self._schedule_processing()
    
    def _schedule_processing(self):
        """Schedule processing with debouncing to avoid multiple triggers"""
        with self.lock:
            self.pending_process = True
            self.last_event_time = time.time()
            
            # Start debounce thread if not already running
            if self.process_thread is None or not self.process_thread.is_alive():
                self.process_thread = Thread(target=self._debounced_process, daemon=True)
                self.process_thread.start()
    
    def _debounced_process(self):
        """Wait for debounce period and then process"""
        while True:
            with self.lock:
                if not self.pending_process:
                    break
                    
                time_since_last_event = time.time() - self.last_event_time
                
                if time_since_last_event >= self.debounce_seconds:
                    self.pending_process = False
                    logger.info(f"Debounce period elapsed, starting automatic processing...")
                    
                    try:
                        result = self.processor.process_all_invoices()
                        
                        if result["success"]:
                            logger.info(f"✓ Auto-processing completed: {result['processed_count']} invoices processed")
                        else:
                            logger.warning(f"⚠ Auto-processing completed with errors: {result['failed_count']} failed")
                    except Exception as e:
                        logger.error(f"Auto-processing failed: {e}")
                    
                    break
            
            # Check again after a short sleep
            time.sleep(1)


class FileWatcherService:
    """Service to monitor incoming folder and trigger automatic processing"""
    
    def __init__(self, incoming_folder: str = "data/incoming", debounce_seconds: int = 5):
        """
        Initialize the file watcher service
        
        Args:
            incoming_folder: Path to the incoming folder to monitor
            debounce_seconds: Time to wait before processing after file changes
        """
        self.incoming_folder = Path(incoming_folder).resolve()
        self.debounce_seconds = debounce_seconds
        self.observer = None
        self.processor = None
        
        # Ensure incoming folder exists
        self.incoming_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File watcher initialized for: {self.incoming_folder}")
    
    def start(self, processor):
        """
        Start watching the incoming folder
        
        Args:
            processor: InvoiceProcessor instance to use for processing
        """
        if self.observer is not None:
            logger.warning("File watcher already running")
            return
        
        self.processor = processor
        
        # Create event handler
        event_handler = InvoiceFileHandler(
            processor=processor,
            debounce_seconds=self.debounce_seconds
        )
        
        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.incoming_folder), recursive=False)
        self.observer.start()
        
        logger.info(f"✓ File watcher started monitoring: {self.incoming_folder}")
        logger.info(f"  - Automatic processing will trigger {self.debounce_seconds}s after new files detected")
    
    def stop(self):
        """Stop watching the incoming folder"""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
            logger.info("File watcher stopped")
    
    def is_running(self) -> bool:
        """Check if the file watcher is running"""
        return self.observer is not None and self.observer.is_alive()