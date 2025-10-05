#!/usr/bin/env python3
"""
Test file for hybrid search functionality
"""
import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.client import WeaviateClient
from tools.hybrid_search import HybridSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_search():
    """Test hybrid search with various scenarios"""
    
    # Read from environment variables
    WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
    COLLECTION_NAME = "Temp"  # Your existing collection
    
    # Ensure URL has proper scheme
    if not WEAVIATE_URL.startswith(('http://', 'https://')):
        WEAVIATE_URL = f"https://{WEAVIATE_URL}"
    
    print("🚀 Starting Hybrid Search Test")
    print("=" * 50)
    print(f"URL: {WEAVIATE_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"API Key: {'Set' if WEAVIATE_API_KEY else 'Not set'}")
    print("=" * 50)
    
    # Initialize client
    client = WeaviateClient(url=WEAVIATE_URL, api_key=WEAVIATE_API_KEY)
    
    try:
        # Test 1: Check if collection exists
        print(f"📋 Checking if collection '{COLLECTION_NAME}' exists...")
        if not client.collection_exists(COLLECTION_NAME):
            print(f"❌ Collection '{COLLECTION_NAME}' does not exist!")
            return
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' exists")
        
        # Test 2: List existing objects
        print(f"\n Listing existing objects in '{COLLECTION_NAME}'...")
        existing_objects = client.list_objects(COLLECTION_NAME, limit=10)
        print(f"Found {len(existing_objects)} objects")
        for i, obj in enumerate(existing_objects[:5]):  # Show first 5
            title = obj.get('properties', {}).get('title', 'No title')
            content = obj.get('properties', {}).get('content', 'No content')
            print(f"  {i+1}. {title}")
            print(f"     Content: {content[:100]}...")
        
        # Test 3: Test keyword-only search using text_search
        print(f"\n Testing keyword-only search (text_search)...")
        test_queries = [
            "machine learning",
            "neural networks", 
            "web development",
            "python programming",
            "data science",
            "artificial intelligence",
            "react javascript",
            "database design",
            "perform calculations",
        ]
        
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            try:
                results = client.text_search(
                    class_name=COLLECTION_NAME,
                    query=query,
                    limit=3
                )
                print(f"  Results: {len(results)} found")
                for i, result in enumerate(results):
                    title = result.get('properties', {}).get('title', 'No title')
                    print(f"    {i+1}. {title}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Test 4: Test hybrid search with keyword only
        print(f"\n🔀 Testing hybrid search with keyword only...")
        for query in test_queries[:4]:  # Test first 4 queries
            print(f"\n  Query: '{query}'")
            try:
                results = client.hybrid_search(
                    class_name=COLLECTION_NAME,
                    query=query,
                    query_vector=None,  # Keyword only
                    alpha=0.7,
                    limit=3
                )
                print(f"  Results: {len(results)} found")
                for i, result in enumerate(results):
                    title = result.get('properties', {}).get('title', 'No title')
                    score = result.get('metadata', {}).get('score', 'N/A')
                    print(f"    {i+1}. {title} (score: {score})")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Test 5: Test with empty query (should fail)
        print(f"\n❌ Testing with empty query (should fail)...")
        try:
            results = client.hybrid_search(
                class_name=COLLECTION_NAME,
                query="",
                query_vector=None,
                alpha=0.7,
                limit=5
            )
            print(f"  Unexpected success: {len(results)} results")
        except Exception as e:
            print(f"  ✅ Expected error: {e}")
        
        # Test 6: Test collection schema
        print(f"\n📊 Testing collection schema...")
        try:
            schema = client.get_collection_schema(COLLECTION_NAME)
            if schema:
                print(f"  Collection name: {schema.get('name')}")
                print(f"  Properties: {len(schema.get('properties', []))}")
                for prop in schema.get('properties', [])[:3]:
                    print(f"    - {prop.get('name')} ({prop.get('data_type')})")
            else:
                print("  ❌ Could not retrieve schema")
        except Exception as e:
            print(f"  ❌ Error getting schema: {e}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test error")
    
    finally:
        client.disconnect()
        print(f"\n✅ Test completed")

if __name__ == "__main__":
    test_hybrid_search()
