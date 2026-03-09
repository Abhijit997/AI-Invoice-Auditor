"""
Test script for Invoice Auditor Chatbot Agent
Tests LangGraph agent with all three tools
"""
from agenticaicapstone.src.agents.chat_agent import create_default_agent


def test_agent():
    """Test the chatbot agent with sample queries"""
    print("\n" + "="*60)
    print("Testing Invoice Auditor Chatbot Agent")
    print("="*60)
    
    # Create agent
    print("\nInitializing agent...")
    try:
        agent = create_default_agent()
        print("✓ Agent created successfully")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        return
    
    # Test queries
    test_queries = [
        {
            "query": "Find invoices for safety equipment",
            "description": "Test invoice search"
        },
        {
            "query": "Show me vendors from Germany",
            "description": "Test vendor search"
        },
        {
            "query": "List products in the Safety category",
            "description": "Test SKU search"
        }
    ]
    
    config = {"configurable": {"thread_id": "test-thread"}}
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"Test {i}: {test['description']}")
        print(f"Query: {test['query']}")
        print(f"{'-'*60}\n")
        
        try:
            messages = [{"role": "user", "content": test["query"]}]
            
            # Stream response
            for event in agent.stream({"messages": messages}, config, stream_mode="values"):
                if event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content"):
                        print(last_message.content)
            
            print("\n✓ Test completed")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    # Check FastAPI
    import requests
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        if response.status_code == 200:
            print("✓ FastAPI backend is running")
        else:
            print("✗ FastAPI returned error status")
            sys.exit(1)
    except Exception as e:
        print(f"✗ FastAPI backend not accessible: {e}")
        print("\nPlease start FastAPI first:")
        print("  uvicorn api.main:app --reload")
        sys.exit(1)
    
    # Run tests
    test_agent()