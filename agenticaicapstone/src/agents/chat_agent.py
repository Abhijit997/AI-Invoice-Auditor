"""
LangGraph Agent for Invoice Auditor Chatbot
Provides conversational interface for searching invoices, vendors, and SKUs
"""
import os
import warnings
from functools import partial
from typing import Optional

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

# Load environment variables BEFORE any LangChain imports
# LangSmith tracing initializes on import and needs LANGCHAIN_* env vars
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Literal

from agenticaicapstone.src.tools.mcp_tools import get_mcp_tools


def assistant(state: MessagesState, sys_msg: SystemMessage, model):
    """
    Assistant node that processes messages and calls LLM with tools
    
    Args:
        state: Current conversation state with message history
        sys_msg: System message with instructions
        model: LLM model with tools bound
    
    Returns:
        Updated state with new message
    """
    # Dynamic trimming of longer messages to stay within token limits
    tokens = 30000
    for i in range(len(state["messages"]) - 1, -1, -1):
        if isinstance(state["messages"][i].content, str):
            limit = int(tokens * 0.7)
            if len(state["messages"][i].content) > limit:
                state["messages"][i].content = (
                    state["messages"][i].content[:limit] + 
                    "...message is too long, truncating rest"
                )
                tokens -= limit
    
    # Invoke model with system message and conversation history
    messages = model.invoke([sys_msg] + state["messages"])
    return {"messages": [messages]}


def reviewer(state: MessagesState, model):
    """
    Reviewer node that validates and formats the final response
    Compares assistant's answer against the original user question
    
    Args:
        state: Current conversation state with message history
        model: LLM model for review
    
    Returns:
        Updated state with reviewed message
    """
    # Get the original user query (last human message)
    user_query = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    
    # Get the assistant's response (last AI message)
    assistant_response = None
    for msg in reversed(state["messages"]):
        if msg.type == "ai" and not msg.tool_calls:
            assistant_response = msg.content
            break
    
    if not user_query or not assistant_response:
        # If we can't find messages, just return as-is
        return {"messages": []}
    
    # Create review prompt
    review_prompt = f"""You are a quality reviewer for an Invoice Auditor Assistant.

Original User Question:
{user_query}

Assistant's Response:
{assistant_response}

Your task:
1. Verify the response fully answers the user's question
2. Check if all requested information is included (amounts, dates, IDs, etc.)
3. Ensure the response is clear, well-formatted, and professional
4. Add any important context or clarifications if missing
5. Fix any formatting issues or unclear statements
6. Try to create tabular scructure whenever possible

Provide the final, polished response that best answers the user's question.
If the response is already excellent, you may return it as-is with minor formatting improvements.
"""
    
    # Invoke reviewer model
    reviewed_message = model.invoke([
        {"role": "system", "content": "You are a quality reviewer that improves responses."},
        {"role": "user", "content": review_prompt}
    ])
    
    # Return the reviewed response
    return {"messages": [AIMessage(content=reviewed_message.content)]}


def build_graph(
    azure_endpoint: str,
    api_key: str,
    model_name: str = "gpt-4o",
    api_version: str = "2024-08-01-preview"
):
    """
    Build LangGraph agent for Invoice Auditor chatbot
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        api_key: Azure OpenAI API key
        model_name: Model deployment name (default: gpt-4o)
        api_version: Azure API version
    
    Returns:
        Compiled LangGraph with checkpointing
    """
    # Initialize the OpenAI LLM with Azure configuration
    model = AzureChatOpenAI(
        model=model_name,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0.1,  # Low temperature for factual responses
    )
    
    # Initialize reviewer model (slightly higher temperature for formatting)
    reviewer_model = AzureChatOpenAI(
        model=model_name,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0.3,
    )
    
    # Get tools from MCP server
    tools = get_mcp_tools()
    model_with_tools = model.bind_tools(tools)
    
    # System prompt for invoice auditor assistant
    system_prompt = """You are an intelligent Invoice Auditor Assistant that helps users search and analyze 
invoice data, vendor information, and product SKUs from approved/production records.

**Your Capabilities:**
- Search invoices by vendor, amount, date, currency, or content using semantic understanding
- Find vendor information including location, contact details, and business descriptions
- Discover products/SKUs by category, description, or specifications
- Answer questions about spending patterns, vendor relationships, and product inventory
- Provide summaries and insights from invoice data

**Available Tools (via MCP Server):**
1. search_invoices_mcp - Search approved invoices with filters for vendor, currency, language
2. search_vendors_mcp - Find vendors by description, country, or currency
3. search_skus_mcp - Search products by category, description, or unit of measure

**Guidelines:**
- Use semantic search to understand user intent (e.g., "safety gear" matches "protective equipment")
- When users ask about spending, use search_invoices_mcp with relevant filters
- For vendor-related questions, use search_vendors_mcp
- For product/item queries, use search_skus_mcp
- Combine multiple tool calls if needed (e.g., search vendor then their invoices)
- If no results found, suggest alternative search terms or filters
- Provide clear summaries with key information (amounts, dates, names)
- Always cite invoice IDs, vendor IDs, or SKU IDs when referencing data
- If data is insufficient, politely inform the user

**Important Notes:**
- All tools connect to MCP server which queries production collections
- Only approved/production data is searchable (not staging data)
- Amounts are in the invoice's currency (check currency field)
- Similarity scores indicate search relevance (higher is better)
- Use filters (country, currency, category) to narrow results

Respond in a professional, concise manner. Focus on answering the user's question with relevant data."""
    
    sys_msg = SystemMessage(content=system_prompt)
    
    # Create assistant node with pre-filled parameters
    assistant_prefilled = partial(assistant, sys_msg=sys_msg, model=model_with_tools)
    reviewer_prefilled = partial(reviewer, model=reviewer_model)
    
    # Routing function - assistant ALWAYS leads to reviewer (after tools if needed)
    def route_after_assistant(state: MessagesState) -> Literal["tools", "reviewer"]:
        """
        Route decision after assistant processes a message:
        - If assistant made tool calls -> execute tools (then loop back to assistant)
        - If no tool calls -> proceed to MANDATORY reviewer (always runs before END)
        
        This ensures ALL responses go through quality review before reaching the user.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"  # Execute tools, then loop back to assistant
        return "reviewer"  # ALWAYS review final response
    
    # Build the graph with MANDATORY reviewer step
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_prefilled)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("reviewer", reviewer_prefilled)  # MANDATORY quality check
    
    # Define edges - reviewer is ALWAYS the final step before END:
    # START -> assistant -> [tools -> assistant (loop)] -> reviewer (MANDATORY) -> END
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"tools": "tools", "reviewer": "reviewer"}  # reviewer is the ONLY path to END
    )
    builder.add_edge("tools", "assistant")  # After tools, loop back to assistant
    builder.add_edge("reviewer", END)  # Reviewer ALWAYS leads to END (no other path)
    
    # Compile with memory checkpointing
    react_graph = builder.compile(checkpointer=MemorySaver())
    return react_graph


def create_default_agent():
    """
    Create agent with Azure OpenAI configuration from .env file
    Loads configuration from environment variables:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_KEY
    - AZURE_OPENAI_DEPLOYMENT
    - AZURE_OPENAI_API_VERSION
    
    Returns:
        Compiled LangGraph agent
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration from environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Validate required environment variables
    if not all([azure_endpoint, api_key, model_name, api_version]):
        raise ValueError(
            "Missing required environment variables. Please ensure .env file contains:\n"
            "  - AZURE_OPENAI_ENDPOINT\n"
            "  - AZURE_OPENAI_KEY\n"
            "  - AZURE_OPENAI_DEPLOYMENT\n"
            "  - AZURE_OPENAI_API_VERSION"
        )
    
    return build_graph(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        model_name=model_name,
        api_version=api_version
    )


# For testing
if __name__ == "__main__":
    agent = create_default_agent()
    
    # Test conversation
    config = {"configurable": {"thread_id": "test-thread"}}
    
    # Test query
    messages = [{"role": "user", "content": "Find invoices for safety equipment"}]
    
    for event in agent.stream({"messages": messages}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()