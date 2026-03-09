"""
Test script for Azure OpenAI text embeddings
"""
from agenticaicapstone.src.utils.azure_openai_client import AzureOpenAIClient
import os
from dotenv import load_dotenv

load_dotenv()

def test_embeddings():
    """Test the embedding functionality"""
    
    print("=" * 70)
    print("Azure OpenAI Embeddings Test - text-embedding-3-large")
    print("=" * 70)
    
    # Check configuration
    print("\n📋 Configuration Check:")
    print("-" * 70)
    embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    embedding_key = os.getenv("AZURE_EMBEDDING_KEY")
    embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    
    print(f"Endpoint: {embedding_endpoint if embedding_endpoint else '❌ NOT SET'}")
    print(f"API Key: {'✓ Set' if embedding_key else '❌ NOT SET'}")
    print(f"Deployment: {embedding_deployment}")
    
    if not embedding_endpoint or not embedding_key:
        print("\n⚠️  ERROR: Missing embedding configuration!")
        print("Please add to your .env file:")
        print("AZURE_EMBEDDING_ENDPOINT=https://openai-poc-abhijit.openai.azure.com/openai/v1/")
        print("AZURE_EMBEDDING_KEY=your-api-key-here")
        print("AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large")
        return
    
    # Initialize client
    print("\n🔧 Initializing Azure OpenAI Client...")
    try:
        client = AzureOpenAIClient()
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Test 1: Single text embedding
    print("\n📝 Test 1: Single Text Embedding")
    print("-" * 70)
    test_text = "This is a test invoice from ABC Company for $1000."
    try:
        embedding = client.create_embeddings(test_text)
        print(f"✓ Text: {test_text}")
        print(f"✓ Embedding dimensions: {len(embedding)}")
        print(f"✓ First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Batch text embeddings
    print("\n📝 Test 2: Batch Text Embeddings")
    print("-" * 70)
    test_texts = [
        "Invoice #001 from Vendor A",
        "Purchase Order for office supplies",
        "Payment receipt for consulting services"
    ]
    try:
        embeddings = client.create_embeddings(test_texts)
        print(f"✓ Number of texts: {len(test_texts)}")
        print(f"✓ Number of embeddings: {len(embeddings)}")
        print(f"✓ Each embedding dimensions: {len(embeddings[0])}")
        for i, text in enumerate(test_texts):
            print(f"  • Text {i+1}: {text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Testing complete")
    print("=" * 70)


if __name__ == "__main__":
    test_embeddings()