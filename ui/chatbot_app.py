"""
Invoice Auditor Chatbot UI
Streamlit interface for conversational invoice/vendor/SKU search
"""
import sys
import os
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# Load environment variables BEFORE any LangChain/agent imports
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import uuid
from datetime import datetime

# Import the agent
from agenticaicapstone.src.agents.chat_agent import create_default_agent


# Page configuration
st.set_page_config(
    page_title="Invoice Auditor Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="chat-header">
    <h1>💬 Invoice Auditor Chatbot</h1>
    <p>Ask questions about invoices, vendors, and products using natural language</p>
</div>
""", unsafe_allow_html=True)


# Initialize session state
if "agent" not in st.session_state:
    try:
        with st.spinner("Initializing AI agent..."):
            st.session_state.agent = create_default_agent()
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.initialized = True
    except ValueError as e:
        st.error("### ⚠️ Configuration Error")
        st.error(str(e))
        st.info("""
        **Please configure your .env file with:**
        - `AZURE_OPENAI_ENDPOINT`
        - `AZURE_OPENAI_KEY`
        - `AZURE_OPENAI_DEPLOYMENT`
        - `AZURE_OPENAI_API_VERSION`
        """)
        st.stop()
    except Exception as e:
        st.error(f"### ❌ Failed to initialize agent: {str(e)}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Actions")
    
    if st.button("🆕 New Conversation", width='stretch'):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Conversation Info")
    st.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    st.divider()
    
    st.markdown("### 💡 Example Queries")
    
    example_queries = [
        "Find all invoices from German vendors",
        "Show me safety equipment purchases",
        "What vendors supply protective gear?",
        "List invoices over 10000 EUR",
        "Find SKUs in the Safety category",
        "Show vendors from USA",
        "What are our recent purchases?",
        "Find helmets and protective equipment"
    ]
    
    for query in example_queries:
        if st.button(f"📝 {query}", width='stretch', key=f"example_{query}"):
            st.session_state.example_query = query
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    
    # Backend status
    import requests
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.markdown("**Configuration:**")
    
    # Show Azure config status
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "Not set")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "Not set")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "Not set")
    
    st.markdown(f"- 🔗 Endpoint: `{endpoint}`")
    st.markdown(f"- ⚡ Model: `{deployment}`")
    st.markdown(f"- 📅 API Version: `{api_version}`")
    st.markdown("- 🔗 LangGraph")
    st.markdown("- 🔍 ChromaDB Vector Search")


# Main chat interface
st.markdown("### 💬 Chat")

# Display info box
st.markdown("""
<div class="info-box">
    <strong>🤖 AI Assistant Ready</strong><br>
    Ask me anything about your invoices, vendors, or products. I can search through approved records 
    and provide insights using semantic understanding.
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example query button clicks
if "example_query" in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                # Prepare messages for agent
                messages_input = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.messages
                ]
                
                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                
                for event in st.session_state.agent.stream(
                    {"messages": messages_input}, 
                    config, 
                    stream_mode="values"
                ):
                    if event["messages"]:
                        last_message = event["messages"][-1]
                        if hasattr(last_message, "content") and last_message.content:
                            full_response = last_message.content
                            response_placeholder.markdown(full_response)
                
                # Add assistant response to history
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_msg = "I encountered an error processing your request. Please try again."
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about invoices, vendors, or products..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                # Prepare messages for agent
                messages_input = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.messages
                ]
                
                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                
                for event in st.session_state.agent.stream(
                    {"messages": messages_input}, 
                    config, 
                    stream_mode="values"
                ):
                    if event["messages"]:
                        last_message = event["messages"][-1]
                        if hasattr(last_message, "content") and last_message.content:
                            full_response = last_message.content
                            response_placeholder.markdown(full_response)
                
                # Add assistant response to history
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_msg = "I encountered an error processing your request. Please try again."
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}\n\nMake sure the FastAPI backend is running at http://localhost:8000"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Invoice Auditor Chatbot • Powered by LangGraph & Azure OpenAI • 
    <a href="http://localhost:8000/docs" target="_blank">API Docs</a></p>
</div>
""", unsafe_allow_html=True)