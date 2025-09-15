#!/usr/bin/env python3
"""
Simple test script for the Deepfake Detection API
"""

import requests
import json
import os
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:8000/')
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False

def test_image_detection():
    """Test image detection endpoint"""
    try:
        # Create test image
        test_image = create_test_image()
        
        # Test with file upload
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post('http://localhost:8000/detect/image', files=files)
        
        print(f"Image Detection: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Image Detection Failed: {e}")
        return False

def test_url_detection():
    """Test URL detection endpoint"""
    try:
        # Test with a public image URL
        test_url = "https://via.placeholder.com/300x200.jpg"
        
        data = {'url': test_url}
        response = requests.post('http://localhost:8000/detect/image', data=data)
        
        print(f"URL Detection: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"URL Detection Failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Deepfake Detection API")
    print("=" * 50)
    
    # Check if backend is running
    print("\n1. Testing Health Check...")
    if not test_health_check():
        print("❌ Backend is not running. Please start it with: python backend/main.py")
        return
    
    print("\n2. Testing Image Detection...")
    test_image_detection()
    
    print("\n3. Testing URL Detection...")
    test_url_detection()
    
    print("\n✅ Tests completed!")

if __name__ == "__main__":
    main()
