#!/usr/bin/env python3
"""
Test script specifically for testing the list_objects function
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.client import WeaviateClient

def test_list_objects():
    """Test the list_objects function with various scenarios"""
    
    # Configuration
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    test_collection = "Temp"  # Use the collection you mentioned
    
    print(f"🔧 Testing list_objects function...")
    print(f"URL: {weaviate_url}")
    print(f"Collection: {test_collection}")
    
    try:
        # Create client
        client = WeaviateClient(weaviate_url, weaviate_api_key)
        connected_client = client.connect()
        
        if not connected_client.is_ready():
            raise Exception("Weaviate client not ready")
        
        print("✅ Connected to Weaviate")
        
        # Test 1: List all objects without any filter
        print("\n📋 Test 1: List all objects (no filter)")
        try:
            objects = client.list_objects(class_name=test_collection)
            print(f"✅ Retrieved {len(objects)} objects")
            if objects:
                print(f"First object: {json.dumps(objects[0], indent=2)}")
            else:
                print("No objects found")
        except Exception as e:
            print(f"❌ Error in Test 1: {e}")
        
        # Test 2: List objects with limit
        print("\n📋 Test 2: List objects with limit=5")
        try:
            objects = client.list_objects(class_name=test_collection, limit=5)
            print(f"✅ Retrieved {len(objects)} objects (limit=5)")
        except Exception as e:
            print(f"❌ Error in Test 2: {e}")
        
        # Test 3: List objects with simple where filter
        print("\n📋 Test 3: List objects with simple where filter")
        try:
            # Try a simple filter - adjust this based on your data
            where_filter = {"name": "test"}  # Change this to match your data
            objects = client.list_objects(class_name=test_collection, where_filter=where_filter)
            print(f"✅ Retrieved {len(objects)} objects with filter {where_filter}")
        except Exception as e:
            print(f"❌ Error in Test 3: {e}")
        
        # Test 4: List objects with Weaviate format filter
        print("\n📋 Test 4: List objects with Weaviate format filter")
        try:
            # Try Weaviate format filter
            where_filter = {
                "path": ["name"],
                "operator": "Equal",
                "valueText": "test"
            }
            objects = client.list_objects(class_name=test_collection, where_filter=where_filter)
            print(f"✅ Retrieved {len(objects)} objects with Weaviate filter")
        except Exception as e:
            print(f"❌ Error in Test 4: {e}")
        
        # Test 5: Check what properties exist in the collection
        print("\n📋 Test 5: Check collection schema")
        try:
            schema = client.get_collection_schema(test_collection)
            print(f"✅ Collection schema: {json.dumps(schema, indent=2)}")
        except Exception as e:
            print(f"❌ Error in Test 5: {e}")
        
        # Test 6: Try to get a single object to see the structure
        print("\n📋 Test 6: Get a single object to see structure")
        try:
            objects = client.list_objects(class_name=test_collection, limit=1)
            if objects:
                print(f"✅ Object structure: {json.dumps(objects[0], indent=2)}")
            else:
                print("No objects to examine")
        except Exception as e:
            print(f"❌ Error in Test 6: {e}")
        
        client.disconnect()
        print("\n✅ All tests completed")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    test_list_objects()
