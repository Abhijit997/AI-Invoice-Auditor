"""
AI Invoice Auditor - Human Review Dashboard
Streamlit UI for reviewing staged invoices before promoting to production.

Run: streamlit run ui/review_app.py
"""
import streamlit as st
import requests
import json
import base64
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
API_BASE_URL = "http://localhost:8000"
PROCESSED_DIR = Path("data/processed")
ACCEPTED_DIR = Path("data/accepted")
REJECTED_DIR = Path("data/rejected")

# Fields that should NOT be editable (system/review fields)
SYSTEM_FIELDS = {
    "review_status", "reviewed_by", "reviewed_at", "review_notes",
    "original_values", "meta_file", "processed_timestamp", "src_invoice_id"
}

# Expected schema definitions for validation
SCHEMA_DEFINITIONS = {
    "invoice": {
        "vendor_name": str,
        "amount": (int, float),
        "currency": str,
        "date": str,
        "po_number": str,
        "language": str,
        "vendor_id": str  # Added by system during processing
    },
    "vendor": {
        "vendor_name": str,
        "country": str,
        "currency": str,
        "full_address": str
    },
    "sku": {
        "category": str,
        "uom": str,
        "gst_rate": (int, float)
    }
}

st.set_page_config(
    page_title="AI Invoice Auditor - Review",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .review-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .review-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .review-header p { color: #b0c4de; margin: 0.3rem 0 0 0; }
    .field-label {
        font-weight: 600;
        color: #1e3a5f;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-title {
        background: #f0f4f8;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #2d5a87;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
        color: #1e3a5f;
    }
    .pending-badge {
        background: #fff3cd;
        color: #856404;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stat-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card h3 { margin: 0; color: #2d5a87; font-size: 1.8rem; }
    .stat-card p { margin: 0; color: #666; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def api_get(endpoint: str):
    """Make GET request to API"""
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API server. Make sure it's running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, data: dict):
    """Make POST request to API"""
    try:
        resp = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API server.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_file_preview(filename: str):
    """Render file preview (PDF or image) in Streamlit"""
    # Try processed folder first, then accepted
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        file_path = ACCEPTED_DIR / filename
    if not file_path.exists():
        file_path = REJECTED_DIR / filename
    if not file_path.exists():
        st.warning(f"File not found: {filename}")
        return
    
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    elif ext in [".png", ".jpg", ".jpeg"]:
        st.image(str(file_path), width='stretch')
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.json(data)
    elif ext == ".csv":
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df, width='stretch', height=700)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            st.text_area("Document Content", value=text, height=700, disabled=True)
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
    else:
        st.info(f"Preview not available for {ext} files")


def render_editable_fields(metadata: dict, prefix: str, record_id: str):
    """Render metadata as key labels (read-only) with editable values.
    Returns dict of changes only."""
    changes = {}
    # Sort keys alphabetically for consistent display order
    for key in sorted(metadata.keys()):
        if key in SYSTEM_FIELDS:
            continue
        value = metadata[key]
        
        unique_key = f"{prefix}_{record_id}_{key}"
        
        col1, col2, col3 = st.columns([2, 5, 1])
        
        with col1:
            st.markdown(f'<div class="field-label">{key}</div>', unsafe_allow_html=True)
        
        with col2:
            # Determine appropriate input type
            if isinstance(value, bool):
                new_val = st.checkbox(key, value=value, key=unique_key, label_visibility="collapsed")
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    new_val = st.number_input(
                        key, value=value, key=unique_key,
                        label_visibility="collapsed", format="%.2f"
                    )
                else:
                    new_val = st.number_input(
                        key, value=value, key=unique_key,
                        label_visibility="collapsed", step=1
                    )
            else:
                str_val = str(value) if value else ""
                new_val = st.text_input(key, value=str_val, key=unique_key, label_visibility="collapsed")
        
        with col3:
            # Remove button for non-required fields
            if st.button("🗑️", key=f"remove_{unique_key}", help="Remove this field"):
                if "removed_fields" not in st.session_state:
                    st.session_state.removed_fields = {}
                if record_id not in st.session_state.removed_fields:
                    st.session_state.removed_fields[record_id] = []
                st.session_state.removed_fields[record_id].append(key)
                st.rerun()
        
        # Track changes
        if str(new_val) != str(value):
            changes[key] = new_val
    
    return changes


def render_add_attribute(prefix: str, record_id: str, schema_type: str, current_metadata: dict):
    """Render UI to add a missing schema attribute."""
    add_key = f"add_attr_{prefix}_{record_id}"
    
    # Get expected fields from schema
    if schema_type not in SCHEMA_DEFINITIONS:
        return
    
    expected_fields = SCHEMA_DEFINITIONS[schema_type]
    
    # Determine which fields are missing
    removed_fields = st.session_state.get("removed_fields", {}).get(record_id, [])
    added_fields = st.session_state.get("added_fields", {}).get(record_id, {})
    
    # Combine current metadata with added fields
    all_current_fields = set(current_metadata.keys()) | set(added_fields.keys())
    # Subtract removed fields
    all_current_fields -= set(removed_fields)
    
    # Find missing schema fields
    missing_fields = [field for field in expected_fields.keys() if field not in all_current_fields and field not in SYSTEM_FIELDS]
    
    if not missing_fields:
        return  # No missing fields to add
    
    with st.expander("➕ Add Missing Schema Attribute", expanded=False):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            selected_field = st.selectbox(
                "Select Attribute", 
                options=missing_fields,
                key=f"{add_key}_field"
            )
        
        with col2:
            if selected_field:
                field_type = expected_fields[selected_field]
                
                # Determine input based on type
                if isinstance(field_type, tuple):
                    # It's (int, float) - use number input
                    new_value = st.number_input(
                        "Value", 
                        key=f"{add_key}_value",
                        value=0.0,
                        format="%.2f"
                    )
                elif field_type == bool:
                    new_value = st.checkbox("Value", key=f"{add_key}_value")
                elif field_type in (int, float):
                    if field_type == float:
                        new_value = st.number_input(
                            "Value",
                            key=f"{add_key}_value",
                            value=0.0,
                            format="%.2f"
                        )
                    else:
                        new_value = st.number_input(
                            "Value",
                            key=f"{add_key}_value",
                            value=0,
                            step=1
                        )
                else:  # str
                    new_value = st.text_input("Value", key=f"{add_key}_value", placeholder="Enter value")
        
        with col3:
            if st.button("Add", key=f"{add_key}_btn", width='stretch'):
                if selected_field:
                    # Validate the value before adding
                    field_type = expected_fields[selected_field]
                    validation_error = None
                    
                    # Type validation
                    if isinstance(field_type, tuple):
                        # It's (int, float) - check if it's a number
                        if not isinstance(new_value, (int, float)):
                            validation_error = f"Expected number, got {type(new_value).__name__}"
                    elif field_type == str:
                        # String validation - check if not empty
                        if not isinstance(new_value, str):
                            validation_error = f"Expected text, got {type(new_value).__name__}"
                        elif not new_value.strip():
                            validation_error = "Value cannot be empty for text fields"
                    elif field_type == int:
                        if not isinstance(new_value, int):
                            validation_error = f"Expected integer, got {type(new_value).__name__}"
                    elif field_type == float:
                        if not isinstance(new_value, (int, float)):
                            validation_error = f"Expected decimal number, got {type(new_value).__name__}"
                    elif field_type == bool:
                        if not isinstance(new_value, bool):
                            validation_error = f"Expected true/false, got {type(new_value).__name__}"
                    
                    if validation_error:
                        st.error(f"❌ Validation Failed: {validation_error}")
                    else:
                        # Add to session state
                        if "added_fields" not in st.session_state:
                            st.session_state.added_fields = {}
                        if record_id not in st.session_state.added_fields:
                            st.session_state.added_fields[record_id] = {}
                        st.session_state.added_fields[record_id][selected_field] = new_value
                        
                        # Remove from removed_fields if it was previously removed
                        if "removed_fields" in st.session_state and record_id in st.session_state.removed_fields:
                            if selected_field in st.session_state.removed_fields[record_id]:
                                st.session_state.removed_fields[record_id].remove(selected_field)
                        
                        st.success(f"✅ Added: {selected_field} = {new_value}")
                        st.rerun()


def render_editable_fields(metadata: dict, prefix: str, record_id: str):
    """Render metadata as key labels (read-only) with editable values.
    Returns dict of changes only."""
    changes = {}
    
    # Apply removed fields filter
    removed_fields = st.session_state.get("removed_fields", {}).get(record_id, [])
    active_metadata = {k: v for k, v in metadata.items() if k not in removed_fields}
    
    # Add custom fields from session state
    added_fields = st.session_state.get("added_fields", {}).get(record_id, {})
    active_metadata.update(added_fields)
    
    # Sort keys alphabetically for consistent display order
    for key in sorted(active_metadata.keys()):
        if key in SYSTEM_FIELDS:
            continue
        value = active_metadata[key]
        
        unique_key = f"{prefix}_{record_id}_{key}"
        
        col1, col2, col3 = st.columns([2, 5, 1])
        
        with col1:
            st.markdown(f'<div class="field-label">{key}</div>', unsafe_allow_html=True)
        
        with col2:
            # Determine appropriate input type
            if isinstance(value, bool):
                new_val = st.checkbox(key, value=value, key=unique_key, label_visibility="collapsed")
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    new_val = st.number_input(
                        key, value=value, key=unique_key,
                        label_visibility="collapsed", format="%.2f"
                    )
                else:
                    new_val = st.number_input(
                        key, value=value, key=unique_key,
                        label_visibility="collapsed", step=1
                    )
            else:
                str_val = str(value) if value else ""
                new_val = st.text_input(key, value=str_val, key=unique_key, label_visibility="collapsed")
        
        with col3:
            # Remove button
            if st.button("🗑️", key=f"remove_{unique_key}", help="Remove this field"):
                if "removed_fields" not in st.session_state:
                    st.session_state.removed_fields = {}
                if record_id not in st.session_state.removed_fields:
                    st.session_state.removed_fields[record_id] = []
                st.session_state.removed_fields[record_id].append(key)
                
                # If this field was previously added (in added_fields), remove it from there too
                if "added_fields" in st.session_state and record_id in st.session_state.added_fields:
                    if key in st.session_state.added_fields[record_id]:
                        del st.session_state.added_fields[record_id][key]
                
                st.rerun()
        
        # Track changes
        if str(new_val) != str(value):
            changes[key] = new_val
    
    return changes


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="review-header">
        <h1>🔍 AI Invoice Auditor — Human Review</h1>
        <p>Review AI-extracted invoice data, edit if needed, then approve or reject</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Sidebar: Stats & Navigation ----
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        stats = api_get("/vector/stats")
        if stats and stats.get("success"):
            cols = stats["collections"]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Staging", cols.get("invoices_stage", 0))
            with c2:
                st.metric("Production", cols.get("invoices", 0))
            
            c3, c4 = st.columns(2)
            with c3:
                st.metric("Vendors (Stage)", cols.get("vendors_stage", 0))
            with c4:
                st.metric("SKUs (Stage)", cols.get("skus_stage", 0))
        
        st.divider()
        st.markdown("### 👤 Reviewer")
        reviewer_name = st.text_input("Your name/email", value="reviewer@company.com", key="reviewer_name")
        
        st.markdown("### 📝 Review Notes")
        review_notes = st.text_area(
            "Notes (optional)",
            placeholder="Add any comments about this review...",
            key="review_notes",
            height=100,
            label_visibility="collapsed"
        )
        
        st.divider()
        st.markdown("### ⚡ Actions")
        approve_clicked = st.button(
            "✅ Approve & Promote",
            type="primary",
            width='stretch',
            key="approve_btn"
        )
        
        reject_clicked = st.button(
            "❌ Reject",
            type="secondary",
            width='stretch',
            key="reject_btn"
        )
        
        skip_clicked = st.button(
            "⏭️ Skip",
            width='stretch',
            key="skip_btn"
        )
        
        # Status area for actions
        status_placeholder = st.empty()
        
        st.divider()
        if st.button("🔄 Refresh", width='stretch'):
            st.rerun()
    
    # ---- Load pending invoices ----
    pending_data = api_get("/review/pending")
    
    if not pending_data or not pending_data.get("success"):
        st.info("Cannot load data. Ensure the API server is running.")
        return
    
    pending_invoices = pending_data.get("invoices", [])
    
    if not pending_invoices:
        st.success("✅ No pending invoices for review. All caught up!")
        
        # Show production stats
        if stats and stats.get("success"):
            cols_data = stats["collections"]
            st.markdown("### Production Collection Stats")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Invoices", cols_data.get("invoices", 0))
            with c2:
                st.metric("Vendors", cols_data.get("vendors", 0))
            with c3:
                st.metric("SKUs", cols_data.get("skus", 0))
        return
    
    # ---- Invoice selector ----
    st.markdown(f"### 📋 Pending Invoices ({len(pending_invoices)})")
    
    invoice_options = {
        f"{inv['invoice_id']} — {inv['vendor_name']} — {inv['currency']} {inv['amount']} — {inv['date']}": inv['invoice_id']
        for inv in pending_invoices
    }
    
    selected_label = st.selectbox(
        "Select an invoice to review:",
        options=list(invoice_options.keys()),
        key="invoice_selector"
    )
    selected_invoice_id = invoice_options[selected_label]
    
    # Clear session state if invoice selection changed to prevent cross-contamination
    if "last_selected_invoice_id" in st.session_state:
        if st.session_state.last_selected_invoice_id != selected_invoice_id:
            # Clear all change tracking when switching invoices
            st.session_state.pop("vendor_changes", None)
            st.session_state.pop("sku_changes", None)
            st.session_state.pop("vendor_content_changes", None)
            st.session_state.pop("sku_content_changes", None)
            st.session_state.pop("added_fields", None)
            st.session_state.pop("removed_fields", None)
    st.session_state.last_selected_invoice_id = selected_invoice_id
    
    st.divider()
    
    # ---- Load full invoice details ----
    details = api_get(f"/review/invoice/{selected_invoice_id}")
    if not details or not details.get("success"):
        st.error("Failed to load invoice details")
        return
    
    invoice = details["invoice"]
    vendors = details.get("vendors", [])
    skus = details.get("skus", [])
    attachment_files = details.get("attachment_files", [])
    meta_file = details.get("meta_file", "")
    
    # ---- Two-column layout: Data | File Preview ----
    col_data, col_preview = st.columns([1, 1], gap="large")
    
    # ======== LEFT COLUMN: Extracted Data (editable) ========
    with col_data:
        st.markdown("### 📝 Extracted Data")
        st.caption("System fields are hidden. Edit values as needed. Use ➕ to add missing schema attributes.")
        
        # -- Invoice Section --
        st.markdown(f'<div class="section-title">📄 Invoice: {invoice["invoice_id"]}</div>', unsafe_allow_html=True)
        invoice_changes = render_editable_fields(
            invoice["metadata"], "inv", invoice["invoice_id"]
        )
        render_add_attribute("inv", invoice["invoice_id"], "invoice", invoice["metadata"])
        
        # -- Invoice Content (editable) --
        with st.expander("📑 Invoice Content (extracted text)", expanded=False):
            st.caption("Edit the content below if needed. A new embedding will be generated on approve.")
            edited_inv_content = st.text_area(
                "Content", value=invoice.get("content", ""),
                height=150, key=f"inv_content_{selected_invoice_id}"
            )
            inv_content_changed = edited_inv_content.strip() != invoice.get("content", "").strip()
        
        # -- Vendors Section --
        if vendors:
            for v_idx, vendor in enumerate(vendors):
                st.markdown(
                    f'<div class="section-title">🏢 Vendor: {vendor["vendor_id"]}</div>',
                    unsafe_allow_html=True
                )
                v_changes = render_editable_fields(
                    vendor["metadata"], f"vend_{v_idx}", vendor["vendor_id"]
                )
                render_add_attribute(f"vend_{v_idx}", vendor["vendor_id"], "vendor", vendor["metadata"])
                # Store changes in session state
                if v_changes:
                    if "vendor_changes" not in st.session_state:
                        st.session_state.vendor_changes = {}
                    st.session_state.vendor_changes[vendor["vendor_id"]] = v_changes
                
                # Vendor content (editable)
                with st.expander(f"📑 Vendor Content: {vendor['vendor_id']}", expanded=False):
                    st.caption("Edit content below. A new embedding will be generated on approve.")
                    edited_v_content = st.text_area(
                        "Vendor Content", value=vendor.get("content", ""),
                        height=120, key=f"vend_content_{v_idx}_{vendor['vendor_id']}",
                        label_visibility="collapsed"
                    )
                    if edited_v_content.strip() != vendor.get("content", "").strip():
                        if "vendor_content_changes" not in st.session_state:
                            st.session_state.vendor_content_changes = {}
                        st.session_state.vendor_content_changes[vendor["vendor_id"]] = edited_v_content
        
        # -- SKUs Section --
        if skus:
            for s_idx, sku in enumerate(skus):
                st.markdown(
                    f'<div class="section-title">📦 SKU: {sku["item_code"]}</div>',
                    unsafe_allow_html=True
                )
                s_changes = render_editable_fields(
                    sku["metadata"], f"sku_{s_idx}", sku["item_code"]
                )
                render_add_attribute(f"sku_{s_idx}", sku["item_code"], "sku", sku["metadata"])
                if s_changes:
                    if "sku_changes" not in st.session_state:
                        st.session_state.sku_changes = {}
                    st.session_state.sku_changes[sku["item_code"]] = s_changes
                
                # SKU content (editable)
                with st.expander(f"📑 SKU Content: {sku['item_code']}", expanded=False):
                    st.caption("Edit content below. A new embedding will be generated on approve.")
                    edited_s_content = st.text_area(
                        "SKU Content", value=sku.get("content", ""),
                        height=120, key=f"sku_content_{s_idx}_{sku['item_code']}",
                        label_visibility="collapsed"
                    )
                    if edited_s_content.strip() != sku.get("content", "").strip():
                        if "sku_content_changes" not in st.session_state:
                            st.session_state.sku_content_changes = {}
                        st.session_state.sku_content_changes[sku["item_code"]] = edited_s_content
        
        # ---- Handle Actions ----
        if approve_clicked:
            reviewer = st.session_state.get("reviewer_name", "reviewer@company.com")
            if not reviewer.strip():
                with status_placeholder.container():
                    st.warning("Please enter your name/email in the sidebar")
            else:
                # Collect all changes including added/removed fields
                action_data = {
                    "reviewed_by": reviewer,
                    "review_notes": review_notes or ""
                }
                
                # Merge invoice changes with added fields and mark removed fields
                complete_invoice_changes = invoice_changes.copy()
                added_inv_fields = st.session_state.get("added_fields", {}).get(invoice["invoice_id"], {})
                # Add from added_fields only if not already changed via text input (text input has priority)
                for key, value in added_inv_fields.items():
                    if key not in complete_invoice_changes:
                        complete_invoice_changes[key] = value
                
                # Mark removed fields for deletion (set to None to signal removal)
                # BUT skip fields that were re-added (present in added_inv_fields)
                removed_inv_fields = st.session_state.get("removed_fields", {}).get(invoice["invoice_id"], [])
                for field in removed_inv_fields:
                    if field not in added_inv_fields:  # Only mark as removed if not re-added
                        complete_invoice_changes[field] = None
                
                if complete_invoice_changes:
                    action_data["invoice_metadata"] = complete_invoice_changes
                
                # Include content changes (triggers re-embedding on server)
                if inv_content_changed:
                    action_data["invoice_content"] = edited_inv_content
                
                # Vendor metadata with added/removed fields
                vendor_edits = st.session_state.get("vendor_changes", {})
                if vendor_edits or st.session_state.get("added_fields") or st.session_state.get("removed_fields"):
                    complete_vendor_edits = {}
                    for vendor in vendors:
                        vid = vendor["vendor_id"]
                        v_changes = vendor_edits.get(vid, {}).copy()
                        added_v_fields = st.session_state.get("added_fields", {}).get(vid, {})
                        # Add from added_fields only if not already changed (text input has priority)
                        for key, value in added_v_fields.items():
                            if key not in v_changes:
                                v_changes[key] = value
                        
                        # Mark removed fields (skip if re-added)
                        removed_v_fields = st.session_state.get("removed_fields", {}).get(vid, [])
                        for field in removed_v_fields:
                            if field not in added_v_fields:  # Only mark as removed if not re-added
                                v_changes[field] = None
                        
                        if v_changes:
                            complete_vendor_edits[vid] = v_changes
                    if complete_vendor_edits:
                        action_data["vendor_metadata"] = complete_vendor_edits
                
                vendor_content_edits = st.session_state.get("vendor_content_changes", {})
                if vendor_content_edits:
                    # Filter to only include vendors that belong to this invoice
                    current_vendor_ids = {v["vendor_id"] for v in vendors}
                    filtered_vendor_content = {vid: content for vid, content in vendor_content_edits.items() if vid in current_vendor_ids}
                    if filtered_vendor_content:
                        action_data["vendor_content"] = filtered_vendor_content
                
                # SKU metadata with added/removed fields
                sku_edits = st.session_state.get("sku_changes", {})
                if sku_edits or st.session_state.get("added_fields") or st.session_state.get("removed_fields"):
                    complete_sku_edits = {}
                    for sku in skus:
                        sku_id = sku["item_code"]
                        s_changes = sku_edits.get(sku_id, {}).copy()
                        added_s_fields = st.session_state.get("added_fields", {}).get(sku_id, {})
                        # Add from added_fields only if not already changed (text input has priority)
                        for key, value in added_s_fields.items():
                            if key not in s_changes:
                                s_changes[key] = value
                        
                        # Mark removed fields (skip if re-added)
                        removed_s_fields = st.session_state.get("removed_fields", {}).get(sku_id, [])
                        for field in removed_s_fields:
                            if field not in added_s_fields:  # Only mark as removed if not re-added
                                s_changes[field] = None
                        
                        if s_changes:
                            complete_sku_edits[sku_id] = s_changes
                    if complete_sku_edits:
                        action_data["sku_metadata"] = complete_sku_edits
                
                sku_content_edits = st.session_state.get("sku_content_changes", {})
                if sku_content_edits:
                    # Filter to only include SKUs that belong to this invoice
                    current_sku_ids = {s["item_code"] for s in skus}
                    filtered_sku_content = {sid: content for sid, content in sku_content_edits.items() if sid in current_sku_ids}
                    if filtered_sku_content:
                        action_data["sku_content"] = filtered_sku_content
                
                with status_placeholder.container():
                    with st.spinner("Approving and promoting to production..."):
                        result = api_post(
                            f"/review/invoice/{selected_invoice_id}/approve",
                            action_data
                        )
                
                if result and result.get("success"):
                    with status_placeholder.container():
                        st.success(f"✅ Invoice {selected_invoice_id} approved and promoted to production!")
                        if result.get("changes_made"):
                            st.info(f"Changes recorded: {json.dumps(result['changes_made'], indent=2)}")
                        if result.get("moved_files"):
                            st.info(f"Files moved to data/accepted: {', '.join(result['moved_files'])}")
                    # Clear session state
                    st.session_state.pop("vendor_changes", None)
                    st.session_state.pop("sku_changes", None)
                    st.session_state.pop("vendor_content_changes", None)
                    st.session_state.pop("sku_content_changes", None)
                    st.session_state.pop("added_fields", None)
                    st.session_state.pop("removed_fields", None)
                    st.rerun()
                else:
                    with status_placeholder.container():
                        st.error("Failed to approve invoice")
        
        if reject_clicked:
            reviewer = st.session_state.get("reviewer_name", "reviewer@company.com")
            if not reviewer.strip():
                with status_placeholder.container():
                    st.warning("Please enter your name/email in the sidebar")
            else:
                action_data = {
                    "reviewed_by": reviewer,
                    "review_notes": review_notes or ""
                }
                
                with status_placeholder.container():
                    with st.spinner("Rejecting invoice..."):
                        result = api_post(
                            f"/review/invoice/{selected_invoice_id}/reject",
                            action_data
                        )
                
                if result and result.get("success"):
                    with status_placeholder.container():
                        st.success(f"❌ Invoice {selected_invoice_id} rejected.")
                        if result.get("moved_files"):
                            st.info(f"Files moved to data/rejected: {', '.join(result['moved_files'])}")
                    st.session_state.pop("vendor_changes", None)
                    st.session_state.pop("sku_changes", None)
                    st.session_state.pop("vendor_content_changes", None)
                    st.session_state.pop("sku_content_changes", None)
                    st.session_state.pop("added_fields", None)
                    st.session_state.pop("removed_fields", None)
                    st.rerun()
                else:
                    with status_placeholder.container():
                        st.error("Failed to reject invoice")
        
        if skip_clicked:
            st.session_state.pop("vendor_changes", None)
            st.session_state.pop("sku_changes", None)
            st.session_state.pop("vendor_content_changes", None)
            st.session_state.pop("sku_content_changes", None)
            st.session_state.pop("added_fields", None)
            st.session_state.pop("removed_fields", None)
            st.rerun()
    
    # ======== RIGHT COLUMN: File Preview ========
    with col_preview:
        st.markdown("### 📄 Source Document")
        
        if attachment_files:
            if len(attachment_files) > 1:
                selected_file = st.selectbox(
                    "Select attachment to preview:",
                    attachment_files,
                    key="file_selector"
                )
            else:
                selected_file = attachment_files[0]
            
            st.caption(f"File: {selected_file}")
            render_file_preview(selected_file)
        else:
            st.info("No attachment files found for preview")
        
        # Show metadata file content
        if meta_file:
            with st.expander("📋 Metadata File Content", expanded=False):
                meta_path = PROCESSED_DIR / meta_file
                if not meta_path.exists():
                    meta_path = ACCEPTED_DIR / meta_file
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        st.json(json.load(f))
                else:
                    st.warning(f"Meta file not found: {meta_file}")


if __name__ == "__main__":
    main()