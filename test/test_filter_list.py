#!/usr/bin/env python3
"""
Test to see how to properly use where filters with Weaviate fetch_objects
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.client import WeaviateClient
from weaviate.classes.query import Filter

def test_filters():
    """Test different ways to use filters with fetch_objects"""
    
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    test_collection = "Temp"
    
    print(f"🔧 Testing filter functionality...")
    
    try:
        client = WeaviateClient(weaviate_url, weaviate_api_key)
        connected_client = client.connect()
        
        if not connected_client.is_ready():
            raise Exception("Weaviate client not ready")
        
        print("✅ Connected to Weaviate")
        
        # Get the collection
        col = connected_client.collections.use(test_collection)
        
        # Test 1: Try using Filter class
        print("\n📋 Test 1: Using Filter.by_property().equal()")
        try:
            # Try to filter by title
            filter_obj = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(where=filter_obj)
            print(f"✅ Retrieved {len(result.objects)} objects with Filter")
            if result.objects:
                print(f"Filtered object: {result.objects[0].properties}")
        except Exception as e:
            print(f"❌ Error with Filter: {e}")
        
        # Test 2: Try with different filter syntax
        print("\n📋 Test 2: Using Filter.by_property().contains_any()")
        try:
            # Try contains filter
            filter_obj = Filter.by_property("title").contains_any(["Machine", "Learning"])
            result = col.query.fetch_objects(where=filter_obj)
            print(f"✅ Retrieved {len(result.objects)} objects with contains filter")
        except Exception as e:
            print(f"❌ Error with contains filter: {e}")
        
        # Test 3: Try with limit and filter
        print("\n📋 Test 3: Using filter with limit")
        try:
            filter_obj = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(where=filter_obj, limit=2)
            print(f"✅ Retrieved {len(result.objects)} objects with filter and limit")
        except Exception as e:
            print(f"❌ Error with filter and limit: {e}")
        
        # Test 4: Try with return_properties and filter
        print("\n📋 Test 4: Using filter with return_properties")
        try:
            filter_obj = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(
                where=filter_obj, 
                return_properties=["title", "content"]
            )
            print(f"✅ Retrieved {len(result.objects)} objects with filter and return_properties")
            if result.objects:
                print(f"Filtered object properties: {result.objects[0].properties}")
        except Exception as e:
            print(f"❌ Error with filter and return_properties: {e}")
        
        # Test 5: Try to see what happens with no matches
        print("\n📋 Test 5: Filter with no matches")
        try:
            filter_obj = Filter.by_property("title").equal("NonExistentTitle")
            result = col.query.fetch_objects(where=filter_obj)
            print(f"✅ Retrieved {len(result.objects)} objects with no-match filter")
        except Exception as e:
            print(f"❌ Error with no-match filter: {e}")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    test_filters()
