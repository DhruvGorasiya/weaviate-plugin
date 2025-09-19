#!/usr/bin/env python3
"""
Test script to check Weaviate connection independently of Dify plugin
"""
import weaviate
from weaviate.classes.init import Auth
from urllib.parse import urlparse

def test_weaviate_connection(url, api_key=None):
    """Test Weaviate connection with different methods"""
    
    print(f"Testing connection to: {url}")
    print(f"API Key: {'Provided' if api_key else 'None'}")
    print("-" * 50)
    
    # Parse URL
    p = urlparse(url)
    host = p.hostname or url.replace("https://", "").replace("http://", "")
    http_secure = (p.scheme == "https")
    http_port = p.port or (443 if http_secure else 80)
    
    auth = Auth.api_key(api_key) if api_key else None
    
    # Test 1: connect_to_custom with different ports
    print("Test 1: connect_to_custom with different ports")
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=host,
            grpc_port=50051,  # Different port
            grpc_secure=http_secure,
            auth_credentials=auth,
        )
        
        if client.is_ready():
            print("✅ connect_to_custom with different ports: SUCCESS")
            client.close()
        else:
            print("❌ connect_to_custom with different ports: FAILED - Not ready")
    except Exception as e:
        print(f"❌ connect_to_custom with different ports: FAILED - {e}")
    
    # Test 2: connect_to_custom with same ports
    print("\nTest 2: connect_to_custom with same ports")
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=host,
            grpc_port=http_port,  # Same port
            grpc_secure=http_secure,
            auth_credentials=auth,
        )
        
        if client.is_ready():
            print("✅ connect_to_custom with same ports: SUCCESS")
            client.close()
        else:
            print("❌ connect_to_custom with same ports: FAILED - Not ready")
    except Exception as e:
        print(f"❌ connect_to_custom with same ports: FAILED - {e}")
    
    # Test 3: connect_to_local
    print("\nTest 3: connect_to_local")
    try:
        client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=50051,
            auth_credentials=auth,
        )
        
        if client.is_ready():
            print("✅ connect_to_local: SUCCESS")
            client.close()
        else:
            print("❌ connect_to_local: FAILED - Not ready")
    except Exception as e:
        print(f"❌ connect_to_local: FAILED - {e}")
    
    # Test 4: connect_to_local without gRPC
    print("\nTest 4: connect_to_local without gRPC")
    try:
        client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            auth_credentials=auth,
        )
        
        if client.is_ready():
            print("✅ connect_to_local without gRPC: SUCCESS")
            client.close()
        else:
            print("❌ connect_to_local without gRPC: FAILED - Not ready")
    except Exception as e:
        print(f"❌ connect_to_local without gRPC: FAILED - {e}")
    
    # Test 5: connect_to_weaviate_cloud
    print("\nTest 5: connect_to_weaviate_cloud")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=auth,
        )
        
        if client.is_ready():
            print("✅ connect_to_weaviate_cloud: SUCCESS")
            client.close()
        else:
            print("❌ connect_to_weaviate_cloud: FAILED - Not ready")
    except Exception as e:
        print(f"❌ connect_to_weaviate_cloud: FAILED - {e}")

if __name__ == "__main__":
    # Test with your Weaviate instance
    url = "https://cyuboxymsbmjsqmsnoggg.c0.us-west3.gcp.weaviate.cloud"
    api_key = "UDNXZUJaYllvRG9PZVlydl90WVJIMkpoWHFzWFZHUGk2dXVWd0VhVWxQaXFEOStZN3dPYnFqQk1KbW9vPV92MjAw"  # Replace with your actual API key
    
    test_weaviate_connection(url, api_key)
