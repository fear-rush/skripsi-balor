"""
Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import EcommerceChatbot

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'bagisto_db',
    'port': 3306
}

# Initialize chatbot
MODEL_PATH = "models/intent_classifier/best_model"
chatbot = EcommerceChatbot(MODEL_PATH, DB_CONFIG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message', '')
    
    if not user_query:
        return jsonify({'error': 'No message provided'}), 400
    
    # Process query
    try:
        result = chatbot.process_query(user_query)
        
        return jsonify({
            'response': result['response'],
            'intent': result['intent'],
            'confidence': round(result['confidence'], 3)
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e),
            'response': 'Maaf, terjadi kesalahan. Coba lagi ya!'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'database': 'connected' if chatbot.query_handler.connection else 'disconnected'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting E-commerce Chatbot Server")
    print("="*70)
    print("\n📱 Open in browser: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("⏹️  Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)