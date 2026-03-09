"""
Reset ChromaDB by completely wiping storage and reinitializing
"""
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from vector_store import InvoiceVectorStore

def reset_chromadb():
    """Completely wipe ChromaDB and reinitialize"""
    print("=" * 60)
    print("CHROMADB COMPLETE RESET")
    print("=" * 60)
    print()
    
    confirm = input("⚠️  This will DELETE ALL data in ChromaDB. Type 'DELETE' to confirm: ")
    if confirm != "DELETE":
        print("❌ Reset cancelled")
        return
    
    print("\n🗑️  Wiping ChromaDB storage...")
    
    # Initialize vector store
    store = InvoiceVectorStore()
    
    # Use the wipe_storage method to completely delete and recreate
    store.wipe_storage()
    
    print("✅ ChromaDB storage completely wiped and reinitialized")
    print("\n📊 New collection stats:")
    stats = store.get_stats()
    for collection, count in stats['collections'].items():
        print(f"  {collection}: {count} documents")
    
    print("\n✨ Done! All collections are empty and ready for fresh data.")

if __name__ == "__main__":
    reset_chromadb()