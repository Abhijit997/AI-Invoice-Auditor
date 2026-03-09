# AI Invoice Auditor - Agentic AI Capstone Project

An end-to-end Agentic AI-powered system for automated multilingual invoice processing, validation, and semantic search using LangGraph, ChromaDB, and Azure OpenAI.

## 🎯 Overview

This system provides intelligent invoice processing with:
- **Automated Processing**: Paired file handling (metadata + attachments)
- **Staging & Review**: Human-in-the-loop approval workflow
- **Vector Search**: ChromaDB with semantic search across approved data
- **MCP Integration**: Single source of truth for search tools
- **Conversational AI**: LangGraph chatbot for natural language queries
- **Observability**: LangSmith tracing for debugging and cost monitoring

---

## 📁 Project Structure

```
AgenticAICapstoneProject/
├── agenticaicapstone/              # Main application package
│   └── src/
│       ├── agents/                 # LangGraph agents (chat_agent.py)
│       ├── tools/                  # MCP tool wrappers (mcp_tools.py)
│       ├── models/                 # Pydantic data models
│       ├── rag/                    # Vector store (vector_store.py, vector_db_loader.py)
│       ├── mcp/                    # MCP server (server.py)
│       └── utils/                  # Azure OpenAI client
│
├── api/                            # FastAPI backend
│   ├── main.py                     # API server
│   ├── processor.py                # Invoice processing logic
│   ├── models.py                   # Data models
│   └── routers/
│       ├── invoices.py             # Invoice endpoints
│       ├── review.py               # Review/approval endpoints
│       └── vector.py               # Vector search endpoints
│
├── ui/                             # Streamlit applications
│   ├── review_app.py               # Human review UI (port 8501)
│   └── chatbot_app.py              # Chatbot UI (port 8502)
│
├── data/                           # Data storage
│   ├── incoming/                   # Incoming invoices (*.meta.json + attachments)
│   ├── processed/                  # Processed invoices - moves to stage collections
│   ├── rejected/                   # Human in the loop - rejected invoices - stays in stage with flag rejected
│   ├── approved/                   # Human in the loop - approved invoices - moves to prod with stage flag approved
│   └── vector_store/               # ChromaDB persistence
│
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp config.sample.env .env
# Edit .env with your Azure OpenAI credentials
# Optional: Add LangSmith API key for observability
```

### 2. Running Applications

Start the following in separate terminals:

```powershell
# Terminal 1 - API Server
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --reload-exclude='data/**' --port 8000

# Terminal 2 - Review App
streamlit run ui/review_app.py

# Terminal 3 - Chatbot App
streamlit run ui/chatbot_app.py --server.port 8502
```

### 3. Start FastAPI Backend

```powershell
# Terminal 1
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --reload-exclude='data/**' --port 8000
```

**Backend provides:**
- Invoice processing API
- Vector search endpoints
- Review/approval workflow
- ERP mock data (vendors, SKUs, POs)

**Verify:** http://localhost:8000/docs

### 4. Process Invoices

```powershell
# Check pending invoices
curl http://localhost:8000/invoices/pending

# Process all invoices (moves to staging)
curl -X POST http://localhost:8000/invoices/process-all
```

### 5. Review & Approve

```powershell
# Terminal 2
streamlit run ui/review_app.py
```

**Access:** http://localhost:8501

- Review staged invoices
- Add/remove schema attributes
- Approve → moves to production collections

### 6. Use Chatbot

```powershell
# Terminal 3
streamlit run ui/chatbot_app.py --server.port 8502
```

**Access:** http://localhost:8502

- Natural language queries
- Semantic search across approved data
- Powered by LangGraph + MCP tools

---

## 🏗️ Architecture

### Data Flow

```
📧 Incoming Invoices (meta.json + PDF/DOCX)
    ↓
🔄 Process API (/invoices/process-all)
    ↓
📝 LLM Agent (extracts structured data) → 🔍 LangSmith Tracing
    ↓
🗄️ Staging Collections (invoices_stage, vendors_stage, skus_stage)
    ↓
👤 Human Review (review_app.py)
    ↓
✅ Approval (/review/invoice/{id}/approve)
    ↓
🎯 Production Collections (invoices, vendors, skus)
    ↓
🤖 Chatbot Search (chat_agent.py → MCP tools → Vector API) → 🔍 LangSmith Tracing
```

### MCP Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          MCP Server (src/mcp/server.py)                     │
│          🎯 TOOL IMPLEMENTATIONS (SINGLE SOURCE)            │
│                                                             │
│  @mcp.tool() search_invoices()                             │
│  @mcp.tool() search_vendors()                              │
│  @mcp.tool() search_skus()                                 │
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
     MCP Protocol│                     │Direct Import
                │                     │
        ┌───────▼───────┐    ┌────────▼────────┐
        │ Claude Desktop│    │  MCP Wrapper     │
        │               │    │  (mcp_tools.py)  │
        └───────────────┘    └────────┬─────────┘
                                      │
                              ┌───────▼────────┐
                              │  LangGraph Agent│
                              │  (chat_agent.py)│
                              └────────────────┘
```

**Benefits:**
- ✅ Single definition - tools defined once, used everywhere
- ✅ Consistency - same behavior for Claude and LangGraph
- ✅ Maintainability - update logic in one place

---

## 🛠️ Technology Stack

- **LLM Framework**: LangGraph, LangChain
- **Observability**: LangSmith (tracing & monitoring)
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector DB**: ChromaDB v0.5.x
- **Embeddings**: Azure OpenAI text-embedding-3-large
- **LLM**: Azure OpenAI gpt-4.1
- **Protocol**: Model Context Protocol (MCP)
- **Python**: 3.13

---

## 📚 Core Components

### 1. Invoice Processing API

**Location:** `api/`

**Key Endpoints:**

```bash
# Health check
GET /

# List pending invoices
GET /invoices/pending

# Process single invoice
POST /invoices/process/{meta_filename}

# Process all invoices
POST /invoices/process-all

# Get statistics
GET /invoices/stats
```

**Metadata Format** (`.meta.json`):
```json
{
  "sender": "billing@example.com",
  "subject": "Invoice INV-001",
  "received_timestamp": "2026-03-01T10:00:00Z",
  "language": "en",
  "attachments": ["INV_001.pdf"]
}
```

**Allowed File Types:** `.pdf`, `.docx`, `.json`, `.csv`, `.jpg`, `.png`

**Processing Logic:**
1. Validates metadata + attachment pairing
2. Extracts content using Azure OpenAI
3. Stores in staging collections for review
4. Removes `attachments` field from metadata

### 2. Vector Store (ChromaDB)

**Location:** `agenticaicapstone/src/rag/vector_store.py`

**Collections:**

| Collection | ID Format | Purpose |
|------------|-----------|---------|
| `invoices` | INV-0000001 | Approved invoices |
| `vendors` | VEND-001 | Approved vendor metadata |
| `skus` | SKU-001 | Approved SKU metadata |
| `invoices_stage` | INV-0000001 | Pending invoices |
| `vendors_stage` | VEND-001 | Pending vendors |
| `skus_stage` | SKU-001 | Pending SKUs |

**Storage:** `data/vector_store/chroma.sqlite3`

**Key Operations:**
```python
from agenticaicapstone.src.rag.vector_store import InvoiceVectorStore

store = InvoiceVectorStore()

# Search
results = store.search_invoices("safety equipment", n_results=5)

# Filter search
results = store.search_invoices(
    "helmets",
    filter_dict={"vendor_name": "Acme Corp"}
)

# Get by ID
invoice = store.get_invoice("INV-0000001")

# Statistics
stats = store.get_stats()
```

### 3. Vector Search API

**Location:** `api/routers/vector.py`

**Production Collection Endpoints:**

```bash
# Search invoices
POST /vector/invoices/search
{
  "query": "safety equipment",
  "n_results": 5,
  "filter_dict": {"vendor_name": "Acme Corp"}
}

# Search vendors
POST /vector/vendors/search
{
  "query": "European suppliers",
  "n_results": 5,
  "filter_dict": {"country": "Germany"}
}

# Search SKUs
POST /vector/skus/search
{
  "query": "protective gear",
  "n_results": 5,
  "filter_dict": {"category": "Safety"}
}

# Get statistics
GET /vector/stats

# Get invoice by ID
GET /vector/invoices/{invoice_id}

# Wipe storage (⚠️ deletes all data)
DELETE /vector/storage/wipe
```

**Staging Collection Endpoints:**
- `POST /vector/invoices-stage/search`
- `POST /vector/vendors-stage/search`
- `POST /vector/skus-stage/search`

### 4. Review & Approval Workflow

**Location:** `ui/review_app.py`, `api/routers/review.py`

**Features:**
- View staged invoices with extracted data
- Side-by-side comparison (extracted vs expected)
- Add missing schema attributes (invoice, vendor, SKU)
- Remove incorrect attributes
- Approve → promotes to production collections

**Approval API:**
```bash
POST /review/invoice/{invoice_id}/approve
{
  "invoice_metadata": {...},
  "vendor_metadata": {...},
  "sku_metadata": {...},
  "added_fields": {"new_field": "value"},
  "removed_fields": ["field_to_delete"]
}
```

### 5. MCP Server

**Location:** `agenticaicapstone/src/mcp/server.py`

**Tools:**
1. `search_invoices(query, n_results, filter_vendor, filter_currency, filter_language)`
2. `search_vendors(query, n_results, filter_country, filter_currency)`
3. `search_skus(query, n_results, filter_category, filter_uom)`

**Start Server:**
```powershell
python -m agenticaicapstone.src.mcp.server
```

**Claude Desktop Config** (`%APPDATA%\Claude\claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "invoice-auditor": {
      "command": "python",
      "args": ["-m", "agenticaicapstone.src.mcp.server"],
      "cwd": "C:\\Users\\abhijit.banerjee\\PycharmProjects\\AgenticAICapstoneProject"
    }
  }
}
```

### 6. LangGraph Chatbot

**Location:** `agenticaicapstone/src/agents/chat_agent.py`, `ui/chatbot_app.py`

**Features:**
- Natural language queries
- Automatic tool selection
- Conversation memory
- Streaming responses

**Example Queries:**
- "Find all invoices from German vendors"
- "Show me purchases over 10,000 EUR"
- "List vendors that supply safety equipment"
- "What protective helmets do we have?"

**Tool Integration:**
- Tools imported from `agenticaicapstone.src.tools.mcp_tools`
- Each tool wraps MCP server function
- Results formatted as text for LLM

### 7. LangSmith Observability

**Location:** Integrated throughout application

**Features:**
- 🔍 **Automatic Tracing**: LangGraph automatically sends traces to LangSmith
- 📊 **Token Usage Tracking**: Monitors input/output tokens for cost analysis
- ⏱️ **Latency Monitoring**: Tracks response times for all LLM calls
- 🐛 **Debugging**: Step-by-step execution traces for agent workflows
- 📈 **Charts & Analytics**: Visual dashboards for token usage and performance

**How It Works:**
1. **Built-in LangGraph Tracing**: When `LANGCHAIN_TRACING_V2=true`, all graph executions are automatically logged
2. **Custom Azure OpenAI Logging**: Manual tracing in `azure_openai_client.py` for direct API calls
3. **Token Format**: Uses LangSmith's expected format with `usage_metadata`:
   ```python
   outputs = {
       "generation": "LLM response text",
       "usage_metadata": {
           "input_tokens": 50,
           "output_tokens": 20,
           "total_tokens": 70
       }
   }
   ```

**Accessing LangSmith:**
- Dashboard: https://smith.langchain.com
- View traces, costs, and performance metrics
- Filter by project: `invoice-auditor`

**What Gets Logged:**
- LangGraph agent executions (nodes, tools, routing)
- LLM calls (prompts, responses, tokens)
- Embeddings generation
- Tool invocations
- Error traces

**SSL Configuration:**
For corporate environments, SSL verification is disabled in:
- `chat_agent.py`
- `api/main.py`
- `ui/chatbot_app.py`
- `azure_openai_client.py`

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Azure Embedding Configuration
AZURE_EMBEDDING_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_EMBEDDING_KEY=your-azure-embedding-api-key-here
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_EMBEDDING_API_VERSION=2023-05-15

# LangSmith Tracing (covers both LangGraph + manual logging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_PROJECT=invoice-auditor

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGSMITH_PROJECT=invoice-auditor
```

### Data Schema

**Invoice Metadata:**
```python
{
  "src_invoice_id": "INV_EN_001",  # Original filename
  "invoice_number": "INV-001",
  "vendor_name": "Acme Corp",
  "vendor_id": "VEND-001",
  "amount": 750.00,
  "currency": "USD",
  "date": "2026-03-01",
  "po_number": "PO-1001",
  "language": "English"
}
```

**Vendor Metadata:**
```python
{
  "src_invoice_id": "INV_EN_001",
  "vendor_name": "Acme Corp",
  "country": "USA",
  "currency": "USD",
  "full_address": "123 Main St, City, State, ZIP"
}
```

**SKU Metadata:**
```python
{
  "src_invoice_id": "INV_EN_001",
  "category": "Safety",
  "uom": "piece",
  "gst_rate": 10
}
```

---

## 🧪 Testing

### Test Vector Store
```bash
python test_vector_store.py
```

### Test API
```bash
# Start server first
python test_api.py
python test_vector_api.py
```

### Test Processing
```bash
python test_processor.py
```

### Test Tools Directly
```python
from agenticaicapstone.src.tools.mcp_tools import search_invoices_mcp

result = search_invoices_mcp.invoke({
    "query": "safety equipment",
    "n_results": 5
})
print(result)
```

**Note:** All tests are traced in LangSmith when `LANGCHAIN_TRACING_V2=true`

---

## 🐛 Troubleshooting

### Issue: Attachments Field Appearing

**Symptom:** `attachments` attribute in collections despite removal code

**Solution:** ChromaDB data contamination from incomplete deletion
```powershell
# Wipe and recreate
curl -X DELETE http://localhost:8000/vector/storage/wipe

# Or use script
python agenticaicapstone/src/rag/reset_chromadb.py
```

### Issue: Approval Error - Array Ambiguity

**Symptom:** "The truth value of an array with more than one element is ambiguous"

**Solution:** Fixed in `vector_store.py` - changed `if embedding:` to `if embedding is not None:`

### Issue: File Lock on Windows

**Symptom:** `[WinError 32]` when wiping ChromaDB

**Solution:**
```powershell
# Stop all servers
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Manually delete and recreate
Remove-Item -Recurse -Force data\vector_store
mkdir data\vector_store
```

### Issue: No Search Results

**Symptom:** Searches return 0 results

**Solutions:**
1. Ensure invoices are **approved** (not just staged)
2. Check production collections have data:
   ```bash
   curl http://localhost:8000/vector/stats
   ```
3. Try broader search terms

### Issue: Import Errors

**Symptom:** `ModuleNotFoundError`

**Solution:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: LangSmith Not Logging

**Symptom:** No traces appearing in LangSmith dashboard

**Solutions:**
1. Verify environment variables:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-key
   LANGCHAIN_PROJECT=invoice-auditor
   ```
2. Check SSL settings (corporate networks may block LangSmith)
3. Verify API key is valid at https://smith.langchain.com
4. Check `.env` file is loaded before LangChain imports
5. View logs for connection errors

### Issue: Token Usage Not Showing

**Symptom:** LangSmith traces don't show token counts

**Solution:** Token format was fixed to use `usage_metadata` structure:
```python
outputs = {
    "generation": "response text",
    "usage_metadata": {
        "input_tokens": 50,
        "output_tokens": 20,
        "total_tokens": 70
    }
}
```
This is now implemented in `azure_openai_client.py`

---

## 📖 API Reference

### Invoice Processing

**GET /invoices/pending**
- Lists pending invoices with validation status
- Shows missing files, invalid extensions

**POST /invoices/process/{meta_filename}**
- Process single invoice pair
- Moves metadata + attachments to processed folder

**POST /invoices/process-all**
- Batch process all valid invoice pairs
- Returns processed and failed counts

**GET /invoices/stats**
- Statistics: total, valid, invalid
- Breakdown by language

### Vector Search

**POST /vector/invoices/search**
```json
{
  "query": "safety equipment",
  "n_results": 5,
  "filter_dict": {"vendor_name": "Acme Corp"}
}
```

**Response:**
```json
{
  "success": true,
  "query": "safety equipment",
  "n_results": 2,
  "results": [
    {
      "rank": 1,
      "invoice_id": "INV-0000001",
      "content": "Invoice text...",
      "metadata": {...},
      "similarity_score": 0.95
    }
  ]
}
```

### Review & Approval

**GET /review/invoices-stage**
- List all staged invoices

**GET /review/invoice-stage/{invoice_id}**
- Get specific staged invoice with metadata

**POST /review/invoice/{invoice_id}/approve**
- Approve and promote to production
- Handles added/removed fields

**DELETE /review/invoice/{invoice_id}/reject**
- Reject and remove from staging

---

## 🎓 Data Schema Verification

### ERP Mock Data Sources

**Files:**
- `api/data/vendors.json` - 6 vendors
- `api/data/sku_master.json` - 11 SKUs
- `api/data/PO Records.json` - 6 purchase orders

**Schema Mapping:**

**Vendor:** `vendors.json` → `VendorDocument`
- vendor_id → vendor_id
- vendor_name → metadata.vendor_name
- country → metadata.country
- currency → metadata.currency

**SKU:** `sku_master.json` → `SKUDocument`
- item_code → item_code
- category → metadata.category
- uom → metadata.uom
- gst_rate → metadata.gst_rate
- *(description from PO Records → content)*

**Invoice:** `processed/*.meta.json` → `InvoiceDocument`
- Extracted by LLM agent
- Validated against schema
- Stored with src_invoice_id reference

---

## 🔐 Security & Production Notes

⚠️ **Development Only** - For production:

1. **API Keys:** Use Azure Key Vault, never hardcode
2. **Authentication:** Implement user login and RBAC
3. **Rate Limiting:** Prevent abuse
4. **Audit Logging:** Track all operations
5. **HTTPS:** Use TLS for all connections
6. **Input Validation:** Sanitize user queries
7. **Data Access Control:** Filter by user role

---

## 📊 Performance

**Typical Response Times:**
- Simple query: ~1-2 seconds
- Single tool call: ~2-4 seconds
- Multiple tool calls: ~4-6 seconds

**Components:**
- LLM inference: ~500-1000ms
- Vector search: ~100-300ms
- Embedding generation: ~200-500ms

**Monitoring:**
- Use LangSmith dashboard to track actual latencies
- Token usage charts show cost breakdown
- Filter by date range for performance trends

**Optimization Tips:**
1. Limit `n_results` to 3-5
2. Use specific queries
3. Add caching for common queries
4. Use faster model for simple queries
5. Monitor token usage in LangSmith to identify expensive operations

---

## 📝 License

Educational Capstone Project