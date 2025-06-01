"""
Debug script to identify the source of the 500 error.
Run this step by step to isolate the issue.
"""

import sys
import traceback
import os
import json

def test_imports():
    """Test 1: Check if all imports work."""
    print("=== Testing Imports ===")
    
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import openai: {e}")
        return False
    
    try:
        # Test your config imports
        sys.path.append('..')
        from config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_LENGTH
        print("✓ Config imported successfully")
        print(f"  - API Key exists: {'Yes' if OPENAI_API_KEY else 'No'}")
        print(f"  - Model: {LLM_MODEL}")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        print("Check if config.py exists and has the required variables")
        return False
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False
    
    return True

def test_openai_client():
    """Test 2: Check if OpenAI client initializes."""
    print("\n=== Testing OpenAI Client ===")
    
    try:
        import openai
        from config import OPENAI_API_KEY, LLM_MODEL
        
        if not OPENAI_API_KEY:
            print("✗ OPENAI_API_KEY is empty")
            return False
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client created successfully")
        
        # Test a simple API call
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✓ OpenAI API call successful")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI client error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_enhanced_client():
    """Test 3: Check if enhanced client works."""
    print("\n=== Testing Enhanced Client ===")
    
    try:
        # Import your enhanced client
        from openai_client import OpenAIClient, ResponseType
        
        client = OpenAIClient()
        print("✓ Enhanced client created successfully")
        
        # Test vague question detection
        is_vague, category = client.analyze_question_specificity("Quel est le capital minimum ?")
        print(f"✓ Vague detection works: {is_vague}, category: {category}")
        
        # Test response generation with empty context
        response = client.generate_response(
            query="Test question",
            context=[],
            language='fr'
        )
        
        print(f"✓ Response generation works: {type(response)}")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced client error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_flask_app():
    """Test 4: Check your Flask app structure."""
    print("\n=== Testing Flask App ===")
    
    # Show current directory and files
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Look for common app files
    app_files = ['app.py', 'main.py', 'server.py', 'run.py']
    found_app = None
    
    for app_file in app_files:
        if os.path.exists(app_file):
            found_app = app_file
            break
    
    if found_app:
        print(f"✓ Found app file: {found_app}")
        
        # Try to read and check the structure
        try:
            with open(found_app, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if '/api/chat' in content:
                print("✓ Found /api/chat endpoint")
            else:
                print("✗ /api/chat endpoint not found")
                
            if '@app.route' in content or '@router' in content:
                print("✓ Found route decorators")
            else:
                print("✗ No route decorators found")
                
        except Exception as e:
            print(f"✗ Error reading app file: {e}")
    else:
        print("✗ No app file found")
    
    return found_app is not None

def test_request_format():
    """Test 5: Check expected request format."""
    print("\n=== Testing Request Format ===")
    
    # Show expected request format
    expected_format = {
        "query": "Quel est le capital minimum ?",
        "language": "fr",
        "context": {}
    }
    
    print("Expected request format:")
    print(json.dumps(expected_format, indent=2))
    
    # Test JSON parsing
    try:
        json_str = json.dumps(expected_format)
        parsed = json.loads(json_str)
        print("✓ JSON parsing works")
        return True
    except Exception as e:
        print(f"✗ JSON parsing error: {e}")
        return False

def create_minimal_working_example():
    """Create a minimal working example."""
    print("\n=== Creating Minimal Working Example ===")
    
    minimal_app = '''
from flask import Flask, request, jsonify
import traceback
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        logging.info("Received request to /api/chat")
        
        # Get request data
        data = request.get_json()
        logging.info(f"Request data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data"}), 400
        
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Simple response for testing
        response = {
            "type": "direct_answer",
            "response": f"Test response for: {query}",
            "context": {"awaiting_clarification": False}
        }
        
        logging.info(f"Sending response: {response}")
        return jsonify({"success": True, "response": response})
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
'''
    
    with open('minimal_test_app.py', 'w') as f:
        f.write(minimal_app)
    
    print("✓ Created minimal_test_app.py")
    print("Run it with: python minimal_test_app.py")
    print("Test with: curl -X POST http://127.0.0.1:5000/api/chat -H 'Content-Type: application/json' -d '{\"query\":\"test\"}'")

def main():
    """Run all tests."""
    print("🚀 Starting Debugging Process...\n")
    
    # Run tests in order
    tests = [
        test_imports,
        test_openai_client,
        test_enhanced_client,
        test_flask_app,
        test_request_format
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if not all(results):
        print("\n🔧 Next steps:")
        if not results[0]:
            print("1. Fix import issues first")
        elif not results[1]:
            print("1. Check OpenAI API key and model settings")
        elif not results[2]:
            print("1. Fix enhanced client implementation")
        elif not results[3]:
            print("1. Check Flask app structure")
        else:
            print("1. Check request format and endpoint logic")
        
        create_minimal_working_example()
    else:
        print("✓ All tests passed! The issue might be in your specific endpoint logic.")

if __name__ == "__main__":
    main()