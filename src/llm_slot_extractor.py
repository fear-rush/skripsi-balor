"""
LLM-based Slot Extractor for Indonesian E-commerce Chatbot

Uses native llama.cpp server (via Homebrew) for semantic slot extraction.
Requires: brew install llama.cpp

Architecture:
    Python Client (this module) --> HTTP --> llama-server --> Qwen3-4B-Instruct-2507

Usage:
    # Start server first:
    # ./scripts/start_llm_server.sh

    from src.llm_slot_extractor import SlotExtractor

    extractor = SlotExtractor()
    result = extractor.extract("stok jaket biru ada ga?")
    print(result)
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class Task(Enum):
    """Supported task types for slot extraction."""
    CHECK_ORDER = "check_order"
    ASK_PRICE = "ask_price"
    CHECK_STOCK = "check_stock"
    ASK_PAYMENT = "ask_payment"
    PRODUCT_INFO = "product_info"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class SlotResult:
    """Structured result from slot extraction."""
    task: str
    entities: Dict[str, Any]
    multi_intent: List[str]
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str]
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "entities": self.entities,
            "multi_intent": self.multi_intent,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "error": self.error
        }


# Valid task types for validation (10 tasks)
VALID_TASKS = {
    'check_order',
    'product_info',
    'product_list',
    'search_product',
    'product_by_category',
    'product_by_attribute',
    'ask_price',
    'check_stock',
    'ask_payment',
    'out_of_scope'
}

# Required fields in response
REQUIRED_FIELDS = {'task', 'entities', 'confidence', 'needs_clarification'}

# System prompt for slot extraction with few-shot examples (10 tasks, full Bahasa Indonesia)
SYSTEM_PROMPT = """
Kamu adalah AI e-commerce chatbot yang bertugas untuk **memahami, mengklasifikasikan, dan mengekstrak intent serta entity dari pertanyaan pengguna** dalam Bahasa Indonesia (formal maupun informal).

- TUGAS UTAMA kamu adalah **intent detection & entity extraction**, BUKAN menjawab pertanyaan.
- Output kamu akan diproses oleh backend (database & business logic).

---

**OUTPUT FORMAT (WAJIB JSON SAJA, TANPA TEKS TAMBAHAN):**

{
  "task": "check_order" |
          "product_info" |
          "product_list" |
          "search_product" |
          "product_by_category" |
          "product_by_attribute" |
          "ask_price" |
          "check_stock" |
          "ask_payment" |
          "out_of_scope",

  "entities": {
    "product_name": "string or null",
    "category": "string or null",
    "attributes": {
      "brand": "string or null",
      "color": "string or null",
      "size": "string or null"
    },
    "order_id": "string or null",
    "quantity": "number or null"
  },

  "multi_intent": ["list of secondary intents if any"],

  "confidence": 0.0 - 1.0,

  "needs_clarification": true | false,

  "clarification_question": "string or null"
}

---

**TASK DEFINITIONS (TOTAL: 10 — SEMUA WAJIB DIPATUHI):**

1. check_order
   - Status pesanan, detail order, status pembayaran, tracking pengiriman
   - order_id WAJIB
   - Jika order_id tidak ada → needs_clarification = true

2. product_info
   - Informasi detail produk (deskripsi, bahan, spesifikasi)

3. product_list
   - Menanyakan produk apa saja yang tersedia secara umum (eksploratif)

4. search_product
   - Mencari produk/kategori via keyword langsung

5. product_by_category
   - Produk berdasarkan kategori tertentu

6. product_by_attribute
   - Produk berdasarkan atribut (brand, warna, ukuran, dll)

7. ask_price
   - Menanyakan harga produk spesifik

8. check_stock
   - Menanyakan ketersediaan stok produk

9. ask_payment
   - Menanyakan metode atau cara pembayaran

10. out_of_scope
    - Di luar cakupan sistem (refund, komplain, jam buka toko, dll)
    - Akan difallback ke admin / WhatsApp

---

NORMALISASI BAHASA INFORMAL:
- aku/gue/gw/lo = saya
- ga/gak/nggak = tidak
- gimana/gmn = bagaimana
- brp/hrg = berapa/harga
- psen/psn = pesanan
- kak = kakak
- dan Bahasa Indonesia informal yang lainnya, mencakup juga typo dan bahasa gaul

---

**CRITICAL RULES:**

1. WAJIB output JSON lengkap
2. DILARANG output teks lain
3. Semua field HARUS ADA (isi null jika tidak dikenal)
4. Jangan mengarang entity yang tidak disebut user
5. Intent ambigu → out_of_scope + confidence 0.5
6. Data penting hilang → needs_clarification = true
7. clarification_question harus singkat & jelas
8. Jangan menjawab sebagai customer service
9. confidence harus mencerminkan kejelasan intent

---

FEW-SHOT EXAMPLES (MENCAKUP SEMUA 10 TASK):

────────────────────────
TASK: ask_payment
User: "bisa bayar pake gopay ga?"
{"task": "ask_payment", "entities": {"product_name": null, "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: check_order
User: "pesanan 12345 udah sampai mana?"
{"task": "check_order", "entities": {"product_name": null, "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": "12345", "quantity": null}, "multi_intent": [], "confidence": 0.95, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: product_info
User: "bahan kaos ini apa?"
{"task": "product_info", "entities": {"product_name": "kaos", "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: product_list
User: "produk ada apa aja?"
{"task": "product_list", "entities": {"product_name": null, "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.85, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: search_product
User: "cari sepatu lari dong"
{"task": "search_product", "entities": {"product_name": "sepatu lari", "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: product_by_category
User: "produk fashion ada apa aja?"
{"task": "product_by_category", "entities": {"product_name": null, "category": "fashion", "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: product_by_attribute
User: "ada baju nike warna merah ukuran L?"
{"task": "product_by_attribute", "entities": {"product_name": "baju", "category": null, "attributes": {"brand": "nike", "color": "merah", "size": "L"}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.95, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: ask_price
User: "harga laptop asus berapa?"
{"task": "ask_price", "entities": {"product_name": "laptop", "category": null, "attributes": {"brand": "asus", "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.95, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: check_stock
User: "stok hp xiaomi masih ada?"
{"task": "check_stock", "entities": {"product_name": "hp", "category": null, "attributes": {"brand": "xiaomi", "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

────────────────────────
TASK: out_of_scope
User: "mau refund barang rusak"
{"task": "out_of_scope", "entities": {"product_name": null, "category": null, "attributes": {"brand": null, "color": null, "size": null}, "order_id": null, "quantity": null}, "multi_intent": [], "confidence": 0.9, "needs_clarification": false, "clarification_question": null}

====================================================================

Sekarang proses input berikut dan outputkan JSON sesuai aturan di atas.
"""


class SlotExtractor:
    """
    LLM-based slot extractor using native llama.cpp server.

    Requires llama-server to be running:
        ./scripts/start_llm_server.sh

    Attributes:
        server_url: URL of the llama-server (default: http://localhost:8080)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: int = 60
    ):
        """
        Initialize the slot extractor.

        Args:
            server_url: URL of running llama-server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.api_url = f"{server_url}/v1/chat/completions"
        self.timeout = timeout

    def is_server_running(self) -> bool:
        """Check if llama-server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _validate_response(self, result: dict) -> bool:
        """
        Validate that response has all required fields and valid task.

        Args:
            result: Parsed JSON response

        Returns:
            True if valid, False otherwise
        """
        if not result or not isinstance(result, dict):
            return False
        if not result.get('task'):
            return False
        if result.get('task') not in VALID_TASKS:
            return False
        if not all(field in result for field in REQUIRED_FIELDS):
            return False
        return True

    def _do_extract(self, query: str, temperature: float) -> tuple:
        """
        Perform single extraction attempt.

        Args:
            query: Customer query
            temperature: Generation temperature

        Returns:
            Tuple of (result_dict, raw_content, error)
        """
        payload = {
            "model": "qwen3-4b-instruct-2507",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            "max_tokens": 512,
            "temperature": temperature,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            raw_content = data['choices'][0]['message']['content']

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(raw_content)

            return (result, raw_content, None)

        except requests.exceptions.ConnectionError:
            return (None, None, "connection_error")
        except json.JSONDecodeError as e:
            return (None, raw_content if 'raw_content' in locals() else None, f"json_error: {e}")
        except Exception as e:
            return (None, None, str(e))

    def extract(self, query: str, temperature: float = 0.1, max_retries: int = 2) -> SlotResult:
        """
        Extract slots from a customer query with automatic retry.

        Args:
            query: Customer query in Indonesian
            temperature: Generation temperature (lower = more deterministic)
            max_retries: Number of retry attempts for invalid responses

        Returns:
            SlotResult with extracted task, entities, etc.
        """
        last_error = None
        last_raw = None

        for attempt in range(max_retries + 1):
            current_temp = min(0.5, temperature + (attempt * 0.1))  # Increase temp on retry

            result, raw_content, error = self._do_extract(query, current_temp)

            # Handle connection error - don't retry
            if error == "connection_error":
                return SlotResult(
                    task="out_of_scope",
                    entities={},
                    multi_intent=[],
                    confidence=0.0,
                    needs_clarification=True,
                    clarification_question="Maaf, sistem sedang tidak tersedia.",
                    error="Server not running. Start with: ./scripts/start_llm_server.sh"
                )

            # Save for error reporting
            if error:
                last_error = error
            if raw_content:
                last_raw = raw_content

            # Check if response is valid
            if result and self._validate_response(result):
                return SlotResult(
                    task=result.get("task", "out_of_scope"),
                    entities=result.get("entities", {}),
                    multi_intent=result.get("multi_intent", []),
                    confidence=result.get("confidence", 0.0),
                    needs_clarification=result.get("needs_clarification", False),
                    clarification_question=result.get("clarification_question"),
                    raw_response=raw_content
                )

            # Log retry attempt (only if not last attempt)
            if attempt < max_retries:
                pass  # Could add logging here

        # All retries failed - return fallback
        return SlotResult(
            task="out_of_scope",
            entities={},
            multi_intent=[],
            confidence=0.0,
            needs_clarification=True,
            clarification_question="Maaf, saya tidak memahami pertanyaan Anda. Bisa diulangi?",
            raw_response=last_raw,
            error=f"extraction_failed_after_{max_retries + 1}_attempts: {last_error}"
        )

    def extract_batch(self, queries: List[str], temperature: float = 0.1) -> List[SlotResult]:
        """
        Extract slots from multiple queries.

        Args:
            queries: List of customer queries
            temperature: Generation temperature

        Returns:
            List of SlotResult objects
        """
        return [self.extract(q, temperature) for q in queries]


# Convenience function
def extract_slots(query: str, server_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """
    Convenience function for one-off slot extraction.

    Args:
        query: Customer query in Indonesian
        server_url: URL of llama-server

    Returns:
        Dictionary with task, entities, etc.
    """
    extractor = SlotExtractor(server_url=server_url)
    result = extractor.extract(query)
    return result.to_dict()


# CLI interface
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="LLM Slot Extractor for Indonesian E-commerce")
    parser.add_argument("query", nargs="?", help="Query to extract slots from")
    parser.add_argument("--server", "-s", default="http://localhost:8080", help="llama-server URL")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    extractor = SlotExtractor(server_url=args.server)

    # Check server
    if not extractor.is_server_running():
        print("ERROR: llama-server is not running!")
        print("")
        print("Start the server first:")
        print("  ./scripts/start_llm_server.sh")
        print("")
        exit(1)

    if args.interactive:
        print("LLM Slot Extractor - Interactive Mode")
        print(f"Server: {args.server}")
        print("Type 'quit' to exit")
        print("-" * 40)

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            start = time.time()
            result = extractor.extract(query)
            elapsed = time.time() - start

            print(f"\nResult ({elapsed:.2f}s):")
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    elif args.query:
        start = time.time()
        result = extractor.extract(args.query)
        elapsed = time.time() - start

        print(f"Query: {args.query}")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Result:")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    else:
        # Demo mode
        print("LLM Slot Extractor - Demo")
        print(f"Server: {args.server}")
        print("=" * 50)

        test_queries = [
            "stok jaket biru ada ga?",
            "harga tas ransel berapa?",
            "pesanan saya 12345 udah sampai mana?",
            "bisa bayar pake gopay ga?",
        ]

        for query in test_queries:
            print(f"\nQuery: \"{query}\"")
            start = time.time()
            result = extractor.extract(query)
            elapsed = time.time() - start

            print(f"  Task: {result.task}")
            print(f"  Entities: {result.entities}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Latency: {elapsed:.2f}s")
