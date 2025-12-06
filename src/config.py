"""
Centralized Configuration for Indonesian E-commerce Chatbot

Environment variables can override defaults:
    LLM_SERVER_URL: URL of the llama.cpp server
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT: Database config
    WHATSAPP_NUMBER: Fallback WhatsApp number
"""

import os


# LLM Server Configuration
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8080")
LLM_MODEL = "Qwen3-4B-Instruct-2507"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Database Configuration (Bagisto MySQL)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "bagisto_db"),
    "port": int(os.getenv("DB_PORT", "3306"))
}

# WhatsApp Fallback
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+6281234567890")

# Confidence Threshold
# If LLM confidence is below this, suggest clarification
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Task Types (matching LLM output - 10 tasks)
# Maps LLM task names to response generator intent names
TASK_TYPES = {
    "check_order": "order_status",
    "product_info": "product_description",
    "product_list": "product_list",
    "search_product": "search_product",
    "product_by_category": "product_by_category",
    "product_by_attribute": "product_by_attribute",
    "ask_price": "product_price",
    "check_stock": "product_stock",
    "ask_payment": "payment_info",
    "out_of_scope": "out_of_scope"
}

# Flask Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
