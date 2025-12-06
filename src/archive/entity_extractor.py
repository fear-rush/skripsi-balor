"""
Semantic Entity Extractor for E-commerce Chatbot
Uses sentence embeddings to match user queries to products in database
Replaces hardcoded keyword matching with dynamic semantic search

Author: Firas
Version: 2.0
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
import mysql.connector
from mysql.connector import Error


class SemanticProductMatcher:
    """
    Extract product entities using semantic similarity
    instead of hardcoded keywords.

    Dynamically fetches products from Bagisto database
    and uses sentence embeddings for matching.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.4
    ):
        """
        Initialize with multilingual sentence transformer

        Args:
            model_name: HuggingFace model for embeddings
            similarity_threshold: Minimum similarity score for a match
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.product_embeddings = None
        self.products = []

        # Lazy load model
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded!")
        except ImportError:
            print("Warning: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def load_products_from_db(self, db_config: Dict) -> bool:
        """
        Load products from Bagisto database and create embeddings

        Args:
            db_config: Database configuration dict

        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            print("Warning: Model not loaded, cannot create embeddings")
            return False

        try:
            print("Connecting to Bagisto database...")
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            # Fetch all active products
            query = """
                SELECT DISTINCT
                    pf.product_id,
                    pf.sku,
                    pf.name,
                    pf.short_description,
                    pf.price
                FROM product_flat pf
                WHERE pf.status = 1
                  AND pf.visible_individually = 1
                  AND pf.name IS NOT NULL
            """

            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                print("Warning: No products found in database")
                cursor.close()
                conn.close()
                return False

            self.products = []
            product_texts = []

            for row in results:
                product = {
                    'id': row['product_id'],
                    'sku': row['sku'],
                    'name': row['name'],
                    'description': row['short_description'] or '',
                    'price': float(row['price']) if row['price'] else 0
                }
                self.products.append(product)

                # Create searchable text: name + keywords from description
                search_text = f"{product['name']} {product['description'][:100]}"
                product_texts.append(search_text)

            cursor.close()
            conn.close()

            # Create embeddings for all products
            print(f"Creating embeddings for {len(self.products)} products...")
            self.product_embeddings = self.model.encode(
                product_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            print("Product embeddings ready!")

            return True

        except Error as e:
            print(f"Database error: {e}")
            return False
        except Exception as e:
            print(f"Error loading products: {e}")
            return False

    def load_products_from_list(self, products: List[Dict]) -> bool:
        """
        Load products from a list (for testing without DB)

        Args:
            products: List of product dicts with 'name' and optional 'description'

        Returns:
            True if successful
        """
        if self.model is None:
            return False

        self.products = products
        product_texts = [
            f"{p.get('name', '')} {p.get('description', '')[:100]}"
            for p in products
        ]

        self.product_embeddings = self.model.encode(
            product_texts,
            normalize_embeddings=True
        )
        return True

    def extract_product(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Extract product entities from user query using semantic similarity

        Args:
            query: User input text
            top_k: Number of top matches to return

        Returns:
            List of matched products with similarity scores
        """
        if self.model is None or self.product_embeddings is None or len(self.products) == 0:
            return []

        # Clean query - remove common question words
        cleaned_query = self._clean_query(query)

        if not cleaned_query:
            return []

        # Get query embedding
        query_embedding = self.model.encode(
            [cleaned_query],
            normalize_embeddings=True
        )[0]

        # Calculate cosine similarity (dot product since normalized)
        similarities = np.dot(self.product_embeddings, query_embedding)

        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        matches = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.similarity_threshold:
                matches.append({
                    'product': self.products[idx],
                    'similarity': score
                })

        return matches

    def _clean_query(self, query: str) -> str:
        """Remove common words that don't help matching"""
        # Indonesian stopwords for product queries
        stopwords = [
            'berapa', 'harga', 'price', 'stok', 'stock', 'ada', 'tidak',
            'gimana', 'bagaimana', 'apakah', 'bisa', 'mau', 'tanya',
            'info', 'detail', 'spek', 'spesifikasi', 'deskripsi',
            'dong', 'kak', 'min', 'gan', 'bro', 'sis', 'ya', 'sih',
            'tolong', 'boleh', 'tau', 'tahu', 'cari', 'minta'
        ]

        query_lower = query.lower()

        # Remove stopwords
        for word in stopwords:
            # Use word boundaries to avoid partial matches
            query_lower = re.sub(r'\b' + word + r'\b', ' ', query_lower)

        # Remove extra spaces
        query_lower = ' '.join(query_lower.split())

        return query_lower.strip()


class EntityExtractor:
    """
    Main entity extraction class combining multiple extractors.
    Handles order IDs, payment methods, and products.
    """

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize entity extractor

        Args:
            db_config: Optional database configuration for loading products
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'bagisto_db',
            'port': 3306
        }

        # Initialize product matcher
        self.product_matcher = SemanticProductMatcher()

        # Load products from database
        if db_config:
            self.product_matcher.load_products_from_db(db_config)

    def load_products(self) -> bool:
        """Load products from database"""
        return self.product_matcher.load_products_from_db(self.db_config)

    def extract_order_id(self, text: str) -> Optional[str]:
        """
        Extract order ID from text using regex patterns

        Args:
            text: User input

        Returns:
            Order ID string or None
        """
        # Patterns for various order ID formats
        patterns = [
            r'#(\d{1,6})',                                    # #12345
            r'(?:order|pesanan|orderan)\s*(?:nomor|no|id)?\s*[#]?(\d{1,6})',  # order 12345
            r'(?:nomor|no)\s*[#]?(\d{1,6})',                  # nomor 12345
            r'ORD[_-]?(\w+)',                                  # ORD-001
            r'\b(\d{4,6})\b',                                  # Standalone 4-6 digit number
        ]

        text_lower = text.lower()

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_payment_method(self, text: str) -> Optional[str]:
        """
        Extract payment method from text

        Args:
            text: User input

        Returns:
            Payment method code or None
        """
        # Payment method keywords mapping
        payment_keywords = {
            'gopay': ['gopay', 'go pay', 'go-pay'],
            'ovo': ['ovo'],
            'dana': ['dana'],
            'shopeepay': ['shopeepay', 'shopee pay', 'spay'],
            'linkaja': ['linkaja', 'link aja'],
            'cod': ['cod', 'cash on delivery', 'bayar di tempat', 'bayar ditempat'],
            'transfer': ['transfer', 'tf', 'transfer bank'],
            'bca': ['bca'],
            'mandiri': ['mandiri'],
            'bni': ['bni'],
            'bri': ['bri'],
            'qris': ['qris'],
            'credit_card': ['kartu kredit', 'credit card', 'cc'],
            'debit': ['kartu debit', 'debit card', 'debit'],
            'va': ['virtual account', 'va'],
        }

        text_lower = text.lower()

        for method, keywords in payment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return method

        return None

    def extract_entities(self, text: str, intent: str) -> Dict:
        """
        Extract all relevant entities based on intent

        Args:
            text: User input
            intent: Classified intent

        Returns:
            Dict of extracted entities
        """
        entities = {}

        if intent == 'order_status':
            order_id = self.extract_order_id(text)
            if order_id:
                entities['order_id'] = order_id

        elif intent == 'payment_info':
            payment_method = self.extract_payment_method(text)
            if payment_method:
                entities['payment_method'] = payment_method

        elif intent in ['product_price', 'product_stock', 'product_description']:
            # Use semantic matching for products
            matches = self.product_matcher.extract_product(text, top_k=3)

            if matches:
                entities['product_matches'] = matches
                # Primary match
                entities['product_name'] = matches[0]['product']['name']
                entities['product_id'] = matches[0]['product']['id']
                entities['match_confidence'] = matches[0]['similarity']

        return entities

    def refresh_products(self) -> bool:
        """Reload products from database (call when products change)"""
        return self.product_matcher.load_products_from_db(self.db_config)


# ============================================================
# TEST / DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTITY EXTRACTOR TEST")
    print("=" * 60)

    # Test without database (using mock products)
    extractor = EntityExtractor()

    # Load mock products for testing
    mock_products = [
        {'id': 1, 'name': 'Arctic Cozy Knit Unisex Beanie', 'description': 'Warm winter beanie'},
        {'id': 2, 'name': 'Arctic Bliss Stylish Winter Scarf', 'description': 'Stylish winter scarf'},
        {'id': 3, 'name': 'Arctic Touchscreen Winter Gloves', 'description': 'Touch screen compatible'},
        {'id': 4, 'name': 'Classic Black Leather Jacket', 'description': 'Premium leather jacket'},
        {'id': 5, 'name': 'Running Shoes Pro', 'description': 'Professional running shoes'},
    ]

    if extractor.product_matcher.load_products_from_list(mock_products):
        print("\nMock products loaded for testing")

    # Test cases
    test_cases = [
        # Order status
        ("cek pesanan 12345", "order_status"),
        ("status order #99", "order_status"),
        ("orderan saya gimana", "order_status"),

        # Payment
        ("bisa bayar pakai gopay?", "payment_info"),
        ("metode pembayaran apa aja", "payment_info"),
        ("terima transfer bank ga", "payment_info"),

        # Product queries
        ("harga beanie berapa", "product_price"),
        ("stok sarung tangan ada ga", "product_stock"),
        ("info produk jacket kulit", "product_description"),
        ("sepatu lari harganya", "product_price"),
    ]

    print("\n" + "-" * 60)
    print("TEST RESULTS:")
    print("-" * 60)

    for text, intent in test_cases:
        entities = extractor.extract_entities(text, intent)
        print(f"\nQuery: \"{text}\"")
        print(f"Intent: {intent}")
        print(f"Entities: {entities}")
