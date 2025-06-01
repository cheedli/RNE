
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
