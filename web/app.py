"""
Flask Web Application for LLM-based E-commerce Chatbot

Uses Qwen3-4B-Instruct-2507 via llama.cpp for semantic slot extraction.

Prerequisites:
    1. Start LLM server: ./scripts/start_llm_server.sh
    2. Ensure Bagisto database is running
    3. Run this app: python web/app.py
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import time
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import LLMChatbot
from src.conversation_context import get_context_store
from src.config import (
    LLM_SERVER_URL,
    DB_CONFIG,
    WHATSAPP_NUMBER,
    LLM_MODEL,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG
)

app = Flask(__name__)

# Initialize LLM chatbot
chatbot = None


def get_chatbot():
    """Get or create chatbot instance."""
    global chatbot
    if chatbot is None:
        chatbot = LLMChatbot(
            llm_server_url=LLM_SERVER_URL,
            db_config=DB_CONFIG,
            whatsapp_number=WHATSAPP_NUMBER
        )
    return chatbot


@app.route('/')
def home():
    """Render chat interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    Process chat message using LLM slot extraction with multi-turn context.

    Request JSON:
        {
            "message": "user query",
            "session_id": "optional-session-id"
        }

    Response JSON:
        {
            "response": "bot response",
            "task": "check_stock|ask_price|...",
            "entities": {"product_name": "...", ...},
            "confidence": 0.95,
            "multi_intent": ["ask_price"],
            "session_id": "session-id",
            "context": {"turn_count": 2, "has_pending": false}
        }
    """
    data = request.json
    user_query = data.get('message', '')
    session_id = data.get('session_id')

    if not user_query:
        return jsonify({'error': 'No message provided'}), 400

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        bot = get_chatbot()
        context_store = get_context_store()

        # Check if LLM is available
        if not bot.is_llm_available():
            return jsonify({
                'response': 'Maaf, sistem AI sedang tidak tersedia. Silakan coba lagi nanti.',
                'task': 'error',
                'entities': {},
                'confidence': 0,
                'multi_intent': [],
                'session_id': session_id,
                'error': 'LLM server not running'
            }), 503

        # Get or create conversation context
        context = context_store.get_or_create(session_id)

        # Process query with context
        start_time = time.time()
        result = bot.process_with_context(user_query, context)
        latency = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            'response': result.response,
            'task': result.task,
            'entities': result.entities,
            'confidence': round(result.confidence, 3),
            'multi_intent': result.multi_intent,
            'needs_clarification': result.needs_clarification,
            'latency_ms': latency,
            'session_id': session_id,
            'context': {
                'turn_count': len(context.history),
                'has_pending_clarification': context.has_pending_clarification(),
                'pending_slot': context.get_pending_slot()
            }
        })

    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': 'Maaf, terjadi kesalahan. Coba lagi ya!',
            'task': 'error',
            'entities': {},
            'confidence': 0,
            'multi_intent': [],
            'session_id': session_id
        }), 500


@app.route('/session/new', methods=['POST'])
def new_session():
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())
    context_store = get_context_store()
    context_store.get_or_create(session_id)
    return jsonify({
        'session_id': session_id,
        'message': 'New session created'
    })


@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session context information."""
    context_store = get_context_store()
    context = context_store.get(session_id)

    if not context:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session_id,
        'context': context.to_dict()
    })


@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete/reset a conversation session."""
    context_store = get_context_store()

    if context_store.delete(session_id):
        return jsonify({
            'message': 'Session deleted',
            'session_id': session_id
        })
    else:
        return jsonify({'error': 'Session not found'}), 404


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for database."""
    bot = get_chatbot()
    db_connected = False

    try:
        if bot.query_handler.connection and bot.query_handler.connection.is_connected():
            db_connected = True
    except:
        pass

    return jsonify({
        'status': 'ok',
        'database': 'connected' if db_connected else 'disconnected'
    })


@app.route('/llm-health', methods=['GET'])
def llm_health():
    """
    LLM server health check with latency measurement.

    Response JSON:
        {
            "status": "online|offline",
            "latency_ms": 45.2,
            "model": "Qwen3-4B-Instruct-2507",
            "server_url": "http://localhost:8080"
        }
    """
    bot = get_chatbot()

    start_time = time.time()
    is_running = bot.is_llm_available()
    latency = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'status': 'online' if is_running else 'offline',
        'latency_ms': latency if is_running else None,
        'model': LLM_MODEL,
        'server_url': LLM_SERVER_URL
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """Return API information."""
    return jsonify({
        'name': 'E-commerce Chatbot API',
        'version': '2.0.0',
        'model': LLM_MODEL,
        'endpoints': {
            'chat': 'POST /chat',
            'health': 'GET /health',
            'llm_health': 'GET /llm-health',
            'info': 'GET /api/info'
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  LLM E-commerce Chatbot Server")
    print("=" * 70)
    print(f"\n  Model: {LLM_MODEL}")
    print(f"  LLM Server: {LLM_SERVER_URL}")
    print(f"  Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("\n" + "-" * 70)
    print(f"\n  Open in browser: http://localhost:{FLASK_PORT}")
    print(f"  Health check: http://localhost:{FLASK_PORT}/health")
    print(f"  LLM status: http://localhost:{FLASK_PORT}/llm-health")
    print("\n  Press Ctrl+C to stop\n")
    print("=" * 70 + "\n")

    # Check LLM on startup
    bot = get_chatbot()
    if bot.is_llm_available():
        print("  LLM Server: ONLINE")
    else:
        print("  LLM Server: OFFLINE")
        print("  Start with: ./scripts/start_llm_server.sh")
    print()

    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
