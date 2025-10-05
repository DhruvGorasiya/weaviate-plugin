#!/usr/bin/env python3
"""
Test to see the correct way to use filters with Weaviate fetch_objects
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

def test_filters_correct():
    """Test the correct way to use filters with fetch_objects"""
    
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    test_collection = "Temp"
    
    print(f"🔧 Testing correct filter functionality...")
    
    try:
        client = WeaviateClient(weaviate_url, weaviate_api_key)
        connected_client = client.connect()
        
        if not connected_client.is_ready():
            raise Exception("Weaviate client not ready")
        
        print("✅ Connected to Weaviate")
        
        # Get the collection
        col = connected_client.collections.use(test_collection)
        
        # Test 1: Using filters parameter (not where)
        print("\n📋 Test 1: Using filters parameter with Filter.by_property().equal()")
        try:
            # Try to filter by title
            filters = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(filters=filters)
            print(f"✅ Retrieved {len(result.objects)} objects with filters")
            if result.objects:
                print(f"Filtered object: {result.objects[0].properties}")
        except Exception as e:
            print(f"❌ Error with filters: {e}")
        
        # Test 2: Try with different filter syntax
        print("\n📋 Test 2: Using Filter.by_property().contains_any()")
        try:
            # Try contains filter
            filters = Filter.by_property("title").contains_any(["Machine", "Learning"])
            result = col.query.fetch_objects(filters=filters)
            print(f"✅ Retrieved {len(result.objects)} objects with contains filter")
        except Exception as e:
            print(f"❌ Error with contains filter: {e}")
        
        # Test 3: Try with limit and filters
        print("\n📋 Test 3: Using filters with limit")
        try:
            filters = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(filters=filters, limit=2)
            print(f"✅ Retrieved {len(result.objects)} objects with filters and limit")
        except Exception as e:
            print(f"❌ Error with filters and limit: {e}")
        
        # Test 4: Try with return_properties and filters
        print("\n📋 Test 4: Using filters with return_properties")
        try:
            filters = Filter.by_property("title").equal("Machine Learning Basics")
            result = col.query.fetch_objects(
                filters=filters, 
                return_properties=["title", "content"]
            )
            print(f"✅ Retrieved {len(result.objects)} objects with filters and return_properties")
            if result.objects:
                print(f"Filtered object properties: {result.objects[0].properties}")
        except Exception as e:
            print(f"❌ Error with filters and return_properties: {e}")
        
        # Test 5: Try to see what happens with no matches
        print("\n📋 Test 5: Filter with no matches")
        try:
            filters = Filter.by_property("title").equal("NonExistentTitle")
            result = col.query.fetch_objects(filters=filters)
            print(f"✅ Retrieved {len(result.objects)} objects with no-match filter")
        except Exception as e:
            print(f"❌ Error with no-match filter: {e}")
        
        # Test 6: Try with multiple conditions (AND)
        print("\n📋 Test 6: Multiple conditions with AND")
        try:
            filters = (
                Filter.by_property("title").contains_any(["Machine"]) &
                Filter.by_property("content").contains_any(["algorithms"])
            )
            result = col.query.fetch_objects(filters=filters)
            print(f"✅ Retrieved {len(result.objects)} objects with AND filter")
        except Exception as e:
            print(f"❌ Error with AND filter: {e}")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    test_filters_correct()
