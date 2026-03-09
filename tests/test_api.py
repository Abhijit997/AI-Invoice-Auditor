"""Test script to verify the API functionality"""
import sys
from pathlib import Path

# Add api directory to path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

# Test imports
try:
    from models import FileInfo, ProcessFileRequest, ProcessFileResponse
    print("✓ Models imported successfully")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

try:
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FastAPI: {e}")
    sys.exit(1)

# Test main module
try:
    import main
    print("✓ Main module imported successfully")
    print(f"✓ App created: {main.app}")
    print(f"✓ Incoming folder: {main.INCOMING_FOLDER}")
    print(f"✓ Processed folder: {main.PROCESSED_FOLDER}")
except Exception as e:
    print(f"✗ Failed to import main: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! The API is ready to run.")
print("\n" + "=" * 60)
print("📋 Available API Endpoints:")
print("=" * 60)

print("\n📧 Invoice Processing (Metadata-Based Paired Processing):")
print("  GET  /invoices/pending         - List pending invoices with validation")
print("  POST /invoices/process/{file}  - Process invoice pair (meta + attachment)")
print("  POST /invoices/process-all     - Process all complete invoice pairs")
print("  GET  /invoices/stats           - Get invoice statistics by language")

print("\n" + "=" * 60)
print("🚀 To start the server, run:")
print("=" * 60)
print("  python api/main.py")
print("  or")
print("  uvicorn api.main:app --reload --port 8000")
print("\n📖 Then visit: http://localhost:8000/docs")
print("=" * 60)