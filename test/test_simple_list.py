#!/usr/bin/env python3
"""
Simple test to see what works with Weaviate fetch_objects
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.client import WeaviateClient

def test_simple():
    """Test the simplest possible fetch_objects call"""
    
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    test_collection = "Temp"
    
    print(f"🔧 Testing simple fetch_objects...")
    
    try:
        client = WeaviateClient(weaviate_url, weaviate_api_key)
        connected_client = client.connect()
        
        if not connected_client.is_ready():
            raise Exception("Weaviate client not ready")
        
        print("✅ Connected to Weaviate")
        
        # Get the collection
        col = connected_client.collections.use(test_collection)
        
        # Test 1: Most basic call
        print("\n📋 Test 1: Basic fetch_objects()")
        try:
            result = col.query.fetch_objects()
            print(f"✅ Retrieved {len(result.objects)} objects")
            if result.objects:
                print(f"First object UUID: {result.objects[0].uuid}")
                print(f"First object properties: {result.objects[0].properties}")
            else:
                print("No objects found in collection")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: With return_properties
        print("\n📋 Test 2: fetch_objects with return_properties=None")
        try:
            result = col.query.fetch_objects(return_properties=None)
            print(f"✅ Retrieved {len(result.objects)} objects")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: With limit
        print("\n📋 Test 3: fetch_objects with limit=5")
        try:
            result = col.query.fetch_objects(limit=5)
            print(f"✅ Retrieved {len(result.objects)} objects")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: Check collection info
        print("\n📋 Test 4: Collection info")
        try:
            config = col.config.get()
            print(f"Collection name: {config.name}")
            print(f"Properties: {[p.name for p in config.properties]}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 5: Check if collection has any data
        print("\n📋 Test 5: Check collection count")
        try:
            # Try to get total count
            agg_result = col.aggregate.over_all(total_count=True)
            total_count = agg_result.total_count
            print(f"Total objects in collection: {total_count}")
        except Exception as e:
            print(f"❌ Error getting count: {e}")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    test_simple()
