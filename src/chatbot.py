"""
Main Chatbot Class
"""

from transformers import pipeline
from src.query_handler import BagistoQueryHandler
from src.response_generator import TemplateResponseGenerator
import re
import json

class EcommerceChatbot:
    
    def __init__(self, model_path, db_config=None):
        print("🤖 Loading chatbot...")
        
        # Load intent classifier
        self.intent_classifier = pipeline(
            "text-classification",
            model=model_path,
            device=-1
        )
        
        # Load label mapping
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            self.label2id = label_mapping['label2id']
            self.id2label = label_mapping['id2label']
        
        # Initialize query handler with database
        self.query_handler = BagistoQueryHandler(db_config)
        
        # Initialize response generator
        self.response_generator = TemplateResponseGenerator()
        
        print("✅ Chatbot ready!")
    
    def extract_order_id(self, text):
        """Extract order ID from text"""
        # Patterns: #12345, ORD001, nomor 12345, order 12345, pesanan 12345
        patterns = [
            r'#?(\d{4,6})',
            r'ORD[_-]?(\w+)',
            r'(?:nomor|order|pesanan)\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_product_name(self, text):
        """Extract product name from text"""
        # Simple keyword extraction
        keywords = ['sepatu', 'laptop', 'kaos', 'tas', 'baju', 'celana', 'hp']
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                return keyword
        
        return None
    
    def process_query(self, user_query):
        """Main chatbot pipeline"""
        
        # Step 1: Intent Classification
        intent_result = self.intent_classifier(user_query)[0]
        intent = intent_result['label']
        confidence = intent_result['score']
        
        print(f"🎯 Intent: {intent} (confidence: {confidence:.3f})")
        
        # Step 2: Entity Extraction
        entities = {}
        
        if intent == 'order_status':
            order_id = self.extract_order_id(user_query)
            if order_id:
                entities['order_id'] = order_id
        
        elif intent == 'product_info':
            product_name = self.extract_product_name(user_query)
            if product_name:
                entities['product_name'] = product_name
        
        print(f"📝 Entities: {entities}")
        
        # Step 3: Query Database
        if intent == 'order_status':
            db_result = self.query_handler.get_order_status(
                order_id=entities.get('order_id')
            )
        
        elif intent == 'payment_info':
            db_result = self.query_handler.get_payment_methods()
        
        elif intent == 'product_info':
            db_result = self.query_handler.get_product_info(
                product_name=entities.get('product_name')
            )
        
        else:
            db_result = {'error': 'Unknown intent'}
        
        print(f"💾 DB Result: {db_result}")
        
        # Step 4: Generate Response
        response = self.response_generator.generate(intent, db_result, user_query)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'response': response
        }


# ============================================================
# TEST CHATBOT (jika dijalankan langsung)
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Check if model path provided
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <model_path>")
        print("Example: python src/chatbot.py models/intent_classifier/best_model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Initialize chatbot
    chatbot = EcommerceChatbot(model_path)
    
    # Test queries
    test_queries = [
        "cek pesanan 12345 dong",
        "metode pembayaran apa aja?",
        "harga laptop berapa?",
        "order 67890 udah sampai belum",
        "bisa bayar pakai gopay?",
        "stock sepatu ada ga",
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING CHATBOT")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"👤 User: {query}")
        result = chatbot.process_query(query)
        print(f"🤖 Bot: {result['response']}")