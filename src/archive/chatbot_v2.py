"""
E-commerce Chatbot V2 - Improved Version

Key Improvements:
1. 6 intents (order_status, payment_info, product_price, product_stock, product_description, out_of_scope)
2. Semantic entity extraction using sentence embeddings (no hardcoded keywords)
3. Confidence thresholding for fallback triggering
4. Optional LLM integration for complex queries
5. WhatsApp redirect for out-of-scope queries

Author: Firas
Version: 2.0
"""

import json
import re
from typing import Dict, Optional, Tuple
from transformers import pipeline

from src.entity_extractor import EntityExtractor
from src.query_handler import BagistoQueryHandler
from src.response_generator_v2 import ResponseGeneratorV2


class EcommerceChatbotV2:
    """
    Improved E-commerce Chatbot with semantic understanding
    and confidence-based fallback mechanism.
    """

    # Default confidence threshold (can be overridden from model config)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70

    def __init__(
        self,
        model_path: str,
        db_config: Optional[Dict] = None,
        use_llm: bool = False,
        llm_backend: str = "llamacpp",
        llm_model: str = "sqlcoder-7b-2",
        whatsapp_number: str = "+6281234567890"
    ):
        """
        Initialize the chatbot

        Args:
            model_path: Path to trained intent classifier model
            db_config: Database configuration dict
            use_llm: Whether to use LLM for complex queries
            llm_backend: LLM backend ("ollama" or "llamacpp")
            llm_model: LLM model name
            whatsapp_number: WhatsApp number for fallback redirect
        """
        print("Loading Chatbot V2...")

        # Load intent classifier
        print("  - Loading intent classifier...")
        self.intent_classifier = pipeline(
            "text-classification",
            model=model_path,
            device=-1  # CPU
        )

        # Load label mapping and threshold
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            self.label2id = label_mapping['label2id']
            self.id2label = label_mapping['id2label']
            # Use threshold from model config if available
            self.confidence_threshold = label_mapping.get(
                'confidence_threshold',
                self.DEFAULT_CONFIDENCE_THRESHOLD
            )

        print(f"  - Confidence threshold: {self.confidence_threshold:.2f}")

        # Initialize database handler
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'bagisto_db',
            'port': 3306
        }
        print("  - Initializing database handler...")
        self.query_handler = BagistoQueryHandler(self.db_config)

        # Initialize semantic entity extractor
        print("  - Loading semantic entity extractor...")
        self.entity_extractor = EntityExtractor(self.db_config)
        self._load_products()

        # Initialize response generator
        self.response_generator = ResponseGeneratorV2(
            whatsapp_number=whatsapp_number
        )

        # Optional LLM integration
        self.use_llm = use_llm
        self.llm_generator = None

        if use_llm:
            try:
                from src.llm_query_generator import LocalLLMQueryGenerator
                print(f"  - Loading LLM ({llm_backend}/{llm_model})...")
                self.llm_generator = LocalLLMQueryGenerator(
                    backend=llm_backend,
                    model=llm_model
                )
            except Exception as e:
                print(f"  - Warning: LLM not available: {e}")
                self.use_llm = False

        print("Chatbot V2 ready!")

    def _load_products(self):
        """Load products for semantic matching"""
        try:
            success = self.entity_extractor.load_products()
            if success:
                print(f"  - Loaded {len(self.entity_extractor.product_matcher.products)} products")
            else:
                print("  - Warning: Could not load products for semantic matching")
        except Exception as e:
            print(f"  - Warning: Error loading products: {e}")

    def refresh_products(self):
        """Refresh product embeddings (call when products change)"""
        return self.entity_extractor.refresh_products()

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify user intent with confidence score

        Args:
            text: User input

        Returns:
            Tuple of (intent, confidence)
        """
        result = self.intent_classifier(text)[0]
        return result['label'], result['score']

    def should_fallback(self, intent: str, confidence: float) -> bool:
        """
        Determine if query should fall back to human support

        Args:
            intent: Classified intent
            confidence: Classification confidence

        Returns:
            True if should fallback
        """
        # Explicit out-of-scope intent
        if intent == 'out_of_scope':
            return True

        # Low confidence on any intent
        if confidence < self.confidence_threshold:
            return True

        return False

    def process_query(self, user_query: str) -> Dict:
        """
        Main chatbot pipeline

        Args:
            user_query: User's question

        Returns:
            Dict with intent, confidence, entities, response
        """
        # Step 1: Intent Classification
        intent, confidence = self.classify_intent(user_query)

        print(f"Intent: {intent} (confidence: {confidence:.3f})")

        # Step 2: Check for fallback
        if self.should_fallback(intent, confidence):
            print(f"Triggering fallback (intent={intent}, conf={confidence:.3f})")
            response = self.response_generator.generate_fallback(
                reason="low_confidence" if intent != 'out_of_scope' else "out_of_scope",
                original_query=user_query
            )
            return {
                'intent': intent,
                'confidence': confidence,
                'entities': {},
                'response': response,
                'fallback': True
            }

        # Step 3: Entity Extraction (using semantic matching)
        entities = self.entity_extractor.extract_entities(user_query, intent)

        print(f"Entities: {entities}")

        # Step 4: Database Query
        db_result = self._query_database(intent, entities, user_query)

        print(f"DB Result: {db_result}")

        # Step 5: Generate Response
        response = self.response_generator.generate(
            intent=intent,
            db_result=db_result,
            user_query=user_query,
            entities=entities
        )

        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'response': response,
            'fallback': False
        }

    def _query_database(self, intent: str, entities: Dict, user_query: str) -> Dict:
        """
        Query database based on intent and entities

        Args:
            intent: Classified intent
            entities: Extracted entities
            user_query: Original query

        Returns:
            Database result dict
        """
        # Order Status
        if intent == 'order_status':
            return self.query_handler.get_order_status(
                order_id=entities.get('order_id')
            )

        # Payment Info
        elif intent == 'payment_info':
            return self.query_handler.get_payment_methods()

        # Product queries - use product_id from semantic matching
        elif intent in ['product_price', 'product_stock', 'product_description']:
            product_id = entities.get('product_id')
            product_name = entities.get('product_name')

            # Determine query type based on intent
            if intent == 'product_price':
                query_type = 'price'
            elif intent == 'product_stock':
                query_type = 'stock'
            else:
                query_type = 'description'

            # Try specific product first
            if product_id:
                result = self.query_handler.get_product_by_id(
                    product_id=product_id,
                    query_type=query_type
                )
                if result.get('found'):
                    return result

            # Try product name search
            if product_name:
                result = self.query_handler.get_product_info(
                    product_name=product_name
                )
                if result.get('found'):
                    return result

            # Try LLM for complex queries
            if self.use_llm and self.llm_generator:
                return self._query_with_llm(user_query, intent)

            # Return featured products as fallback
            return self.query_handler.get_product_info(product_name=None)

        # Fallback for unknown intents
        else:
            return {'error': f'Unknown intent: {intent}'}

    def _query_with_llm(self, user_query: str, intent: str) -> Dict:
        """
        Use LLM to generate and execute SQL query

        Args:
            user_query: User's question
            intent: Classified intent

        Returns:
            Query result dict
        """
        if not self.llm_generator:
            return {'error': 'LLM not available'}

        try:
            sql, is_safe = self.llm_generator.generate_sql(user_query, intent)

            if is_safe and sql:
                result = self.llm_generator.execute_safe_query(
                    sql,
                    self.query_handler
                )
                if result and result.get('success'):
                    return {
                        'found': True,
                        'source': 'llm',
                        'products': result['data'],
                        'sql': result.get('sql')
                    }

        except Exception as e:
            print(f"LLM query error: {e}")

        return {'found': False, 'error': 'LLM query failed'}


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Command-line interface for testing the chatbot"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.chatbot_v2 <model_path>")
        print("Example: python -m src.chatbot_v2 models/intent_classifier_v2/best_model")
        print("\nOptions:")
        print("  --use-llm       Enable LLM for complex queries")
        print("  --threshold N   Set confidence threshold (0.0-1.0)")
        sys.exit(1)

    model_path = sys.argv[1]

    # Parse optional arguments
    use_llm = '--use-llm' in sys.argv
    threshold = None

    for i, arg in enumerate(sys.argv):
        if arg == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])

    # Initialize chatbot
    chatbot = EcommerceChatbotV2(
        model_path=model_path,
        use_llm=use_llm
    )

    # Override threshold if specified
    if threshold:
        chatbot.confidence_threshold = threshold
        print(f"Threshold set to: {threshold:.2f}")

    # Test queries
    test_queries = [
        # Product queries
        "berapa harga arctic beanie?",
        "stok scarf masih ada ga?",
        "deskripsi gloves dong",

        # Order queries
        "cek pesanan 12345",
        "status order saya gimana",

        # Payment
        "bisa bayar pakai gopay?",
        "metode pembayaran apa aja",

        # Noisy queries
        "brp hrg benie artik",
        "stok srf msh ad g",

        # Out of scope
        "apakah bisa diantar ke mars?",
        "siapa presiden indonesia?",
    ]

    print("\n" + "=" * 70)
    print("CHATBOT V2 TEST")
    print("=" * 70)

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"User: {query}")
        result = chatbot.process_query(query)
        print(f"\nBot: {result['response']}")
        print(f"\n[Intent: {result['intent']}, Confidence: {result['confidence']:.3f}, Fallback: {result['fallback']}]")


if __name__ == "__main__":
    main()
