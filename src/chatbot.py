"""
LLM-based E-commerce Chatbot for Bagisto

Uses Qwen3-4B-Instruct-2507 via llama.cpp for semantic slot extraction,
replacing the traditional intent classifier approach.

Architecture:
    User Query → SlotExtractor → Database Query → Response Generator

Usage:
    from src.chatbot import LLMChatbot
    from src.config import LLM_SERVER_URL, DB_CONFIG, WHATSAPP_NUMBER

    chatbot = LLMChatbot(
        llm_server_url=LLM_SERVER_URL,
        db_config=DB_CONFIG,
        whatsapp_number=WHATSAPP_NUMBER
    )

    result = chatbot.process("stok sepatu nike size 42 ada?")
    print(result.response)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging

from src.llm_slot_extractor import SlotExtractor, SlotResult
from src.query_handler import BagistoQueryHandler
from src.response_generator import ResponseGeneratorV2
from src.config import CONFIDENCE_THRESHOLD, TASK_TYPES
from src.conversation_context import (
    ConversationContext,
    ContextStore,
    get_context_store,
    determine_clarification_slot,
    CLARIFICATION_SLOT_MAP
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Structured response from chatbot."""
    response: str
    task: str
    entities: Dict[str, Any]
    confidence: float
    multi_intent: List[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    db_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "response": self.response,
            "task": self.task,
            "entities": self.entities,
            "confidence": self.confidence,
            "multi_intent": self.multi_intent,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "error": self.error
        }


class LLMChatbot:
    """
    LLM-based chatbot using semantic slot extraction.

    This chatbot uses a local LLM (Qwen3-4B-Instruct-2507) via llama.cpp
    to extract structured information from customer queries, then queries
    the Bagisto database and generates appropriate responses.

    Attributes:
        slot_extractor: LLM-based slot extraction module
        query_handler: Bagisto database query handler
        response_generator: Template-based response generator
        whatsapp_number: Fallback WhatsApp number for out-of-scope queries
    """

    def __init__(
        self,
        llm_server_url: str = "http://localhost:8080",
        db_config: Optional[Dict[str, Any]] = None,
        whatsapp_number: str = "+6281234567890"
    ):
        """
        Initialize the LLM chatbot.

        Args:
            llm_server_url: URL of the llama.cpp server
            db_config: Database configuration dictionary
            whatsapp_number: WhatsApp number for human fallback
        """
        self.slot_extractor = SlotExtractor(server_url=llm_server_url)
        self.query_handler = BagistoQueryHandler(db_config=db_config)
        self.response_generator = ResponseGeneratorV2(whatsapp_number=whatsapp_number)
        self.whatsapp_number = whatsapp_number

        logger.info(f"LLMChatbot initialized with server: {llm_server_url}")

    def is_llm_available(self) -> bool:
        """Check if the LLM server is running."""
        return self.slot_extractor.is_server_running()

    def process(self, query: str) -> ChatResponse:
        """
        Process a customer query and return a response.

        Args:
            query: Customer query in Indonesian

        Returns:
            ChatResponse with response text and metadata
        """
        logger.info(f"Processing query: {query}")

        # Step 1: Extract slots using LLM
        try:
            slots = self.slot_extractor.extract(query)
        except Exception as e:
            logger.error(f"Slot extraction failed: {e}")
            return self._error_response(
                "Maaf, sistem sedang tidak tersedia. Silakan coba lagi.",
                str(e)
            )

        logger.info(f"Extracted: task={slots.task}, entities={slots.entities}")

        # Step 2: Handle clarification if needed
        if slots.needs_clarification or slots.confidence < CONFIDENCE_THRESHOLD:
            return self._handle_clarification(slots)

        # Step 3: Handle out-of-scope queries
        if slots.task == "out_of_scope":
            return self._handle_out_of_scope(slots)

        # Step 4: Query database based on task and entities
        db_result = self._query_database(slots)

        # Step 5: Generate response
        response = self._generate_response(slots, db_result)

        return ChatResponse(
            response=response,
            task=slots.task,
            entities=slots.entities,
            confidence=slots.confidence,
            multi_intent=slots.multi_intent,
            db_result=db_result
        )

    def process_with_context(
        self,
        query: str,
        context: ConversationContext
    ) -> ChatResponse:
        """
        Process a query with conversation context for multi-turn support.

        This method handles:
        1. Pending clarifications (e.g., user provides order_id after being asked)
        2. Entity carryover from previous turns
        3. Conversation history tracking

        Args:
            query: User's query
            context: ConversationContext for this session

        Returns:
            ChatResponse with response and metadata
        """
        logger.info(f"Processing with context: {context.get_conversation_summary()}")

        # Step 1: Check if this is a clarification response
        if context.has_pending_clarification():
            resolution = context.resolve_clarification_input(query)

            if resolution.get("resolved"):
                logger.info(f"Resolved clarification: {resolution['slot_name']}={resolution['slot_value']}")

                # Create a pseudo SlotResult with resolved entities
                task = resolution["task"]
                entities = resolution["entities"]

                # Query database with resolved entities
                class ResolvedSlots:
                    def __init__(self, t, e):
                        self.task = t
                        self.entities = e
                        self.confidence = 0.95
                        self.multi_intent = []
                        self.needs_clarification = False
                        self.clarification_question = None

                slots = ResolvedSlots(task, entities)
                db_result = self._query_database(slots)
                response = self._generate_response(slots, db_result)

                # Record this turn (clarification resolved)
                context.add_turn(
                    user_query=query,
                    bot_response=response,
                    task=task,
                    entities=entities,
                    confidence=0.95,
                    needs_clarification=False
                )

                return ChatResponse(
                    response=response,
                    task=task,
                    entities=entities,
                    confidence=0.95,
                    multi_intent=[],
                    db_result=db_result
                )

        # Step 2: Normal processing with LLM
        try:
            slots = self.slot_extractor.extract(query)
        except Exception as e:
            logger.error(f"Slot extraction failed: {e}")
            return self._error_response(
                "Maaf, sistem sedang tidak tersedia. Silakan coba lagi.",
                str(e)
            )

        logger.info(f"Extracted: task={slots.task}, entities={slots.entities}")

        # Step 3: Merge with active entities from context (entity carryover)
        merged_entities = self._merge_entities(context.active_entities, slots.entities)
        slots.entities = merged_entities

        # Step 4: Handle clarification if needed
        if slots.needs_clarification or slots.confidence < CONFIDENCE_THRESHOLD:
            # Determine which slot needs clarification
            clarification_slot = determine_clarification_slot(slots.task, slots.entities)

            response = self._handle_clarification(slots)

            # Record turn and set pending clarification
            context.add_turn(
                user_query=query,
                bot_response=response.response,
                task=slots.task,
                entities=slots.entities,
                confidence=slots.confidence,
                needs_clarification=True,
                clarification_slot=clarification_slot
            )

            return response

        # Step 5: Handle out-of-scope queries
        if slots.task == "out_of_scope":
            response = self._handle_out_of_scope(slots)

            context.add_turn(
                user_query=query,
                bot_response=response.response,
                task=slots.task,
                entities=slots.entities,
                confidence=slots.confidence
            )

            return response

        # Step 6: Query database
        db_result = self._query_database(slots)

        # Step 7: Check if we need clarification based on missing required entities
        if "error" in db_result and db_result.get("error") in ["no_order_id", "no_product", "no_keyword"]:
            clarification_slot = CLARIFICATION_SLOT_MAP.get(slots.task)
            clarification_msg = self._get_clarification_message(slots.task, db_result)

            context.add_turn(
                user_query=query,
                bot_response=clarification_msg,
                task=slots.task,
                entities=slots.entities,
                confidence=slots.confidence,
                needs_clarification=True,
                clarification_slot=clarification_slot
            )

            return ChatResponse(
                response=clarification_msg,
                task=slots.task,
                entities=slots.entities,
                confidence=slots.confidence,
                multi_intent=slots.multi_intent,
                needs_clarification=True,
                clarification_question=clarification_msg
            )

        # Step 8: Generate response
        response = self._generate_response(slots, db_result)

        # Record turn
        context.add_turn(
            user_query=query,
            bot_response=response,
            task=slots.task,
            entities=slots.entities,
            confidence=slots.confidence
        )

        return ChatResponse(
            response=response,
            task=slots.task,
            entities=slots.entities,
            confidence=slots.confidence,
            multi_intent=slots.multi_intent,
            db_result=db_result
        )

    def _merge_entities(
        self,
        active_entities: Dict[str, Any],
        new_entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge active entities from context with newly extracted entities.

        New entities take precedence over active ones.
        """
        merged = {}

        # Start with active entities
        for key, value in active_entities.items():
            if value is not None and value != "" and value != {}:
                merged[key] = value

        # Override with new entities
        for key, value in new_entities.items():
            if value is not None and value != "" and value != {}:
                if key == "attributes" and isinstance(value, dict):
                    if "attributes" not in merged:
                        merged["attributes"] = {}
                    for attr_key, attr_value in value.items():
                        if attr_value is not None:
                            merged["attributes"][attr_key] = attr_value
                else:
                    merged[key] = value

        return merged

    def _get_clarification_message(self, task: str, db_result: Dict[str, Any]) -> str:
        """Get appropriate clarification message based on task."""
        messages = {
            "check_order": "Mohon berikan nomor pesanan Anda agar saya bisa cek statusnya.",
            "check_stock": "Produk apa yang ingin dicek stoknya?",
            "ask_price": "Produk mana yang ingin dicek harganya?",
            "product_info": "Produk mana yang ingin diketahui informasinya?",
            "search_product": "Produk apa yang ingin dicari?",
            "product_by_category": "Kategori apa yang ingin dilihat?",
            "product_by_attribute": "Produk atau atribut apa yang dicari?",
        }
        return messages.get(task, db_result.get("message", "Bisa diperjelas pertanyaannya?"))

    def _handle_clarification(self, slots: SlotResult) -> ChatResponse:
        """Handle queries that need clarification."""
        clarification = slots.clarification_question or \
            "Maaf, bisa tolong diperjelas pertanyaannya?"

        return ChatResponse(
            response=clarification,
            task=slots.task,
            entities=slots.entities,
            confidence=slots.confidence,
            multi_intent=slots.multi_intent,
            needs_clarification=True,
            clarification_question=clarification
        )

    def _handle_out_of_scope(self, slots: SlotResult) -> ChatResponse:
        """Handle out-of-scope queries with WhatsApp fallback."""
        response = self.response_generator.generate_fallback(reason="out_of_scope")

        return ChatResponse(
            response=response,
            task="out_of_scope",
            entities=slots.entities,
            confidence=slots.confidence,
            multi_intent=slots.multi_intent
        )

    def _query_database(self, slots: SlotResult) -> Dict[str, Any]:
        """
        Query the database based on extracted task and entities.

        Args:
            slots: Extracted slot information

        Returns:
            Dictionary with query results
        """
        entities = slots.entities
        task = slots.task

        try:
            if task == "check_order":
                return self._query_order(entities)
            elif task == "check_stock":
                return self._query_stock(entities)
            elif task == "ask_price":
                return self._query_price(entities)
            elif task == "ask_payment":
                return self._query_payment(entities)
            elif task == "product_info":
                return self._query_product_info(entities)
            elif task == "product_list":
                return self._query_product_list(entities)
            elif task == "search_product":
                return self._query_search_product(entities)
            elif task == "product_by_category":
                return self._query_product_by_category(entities)
            elif task == "product_by_attribute":
                return self._query_product_by_attribute(entities)
            else:
                return {"error": f"Unknown task: {task}"}

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {"error": str(e)}

    def _query_order(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query order status."""
        order_id = entities.get("order_id")

        if not order_id:
            return {
                "error": "no_order_id",
                "message": "Nomor pesanan tidak ditemukan dalam pertanyaan"
            }

        # query_handler.get_order_status returns:
        # {'found': True, 'order': {...}} or {'found': False, 'order_id': ...}
        order_result = self.query_handler.get_order_status(order_id)

        # Handle error from query_handler
        if order_result.get('error'):
            return order_result

        # Handle order not found
        if not order_result.get('found'):
            return {
                "error": "order_not_found",
                "message": f"Pesanan {order_id} tidak ditemukan",
                "order_id": order_id
            }

        # Order found - add items to the order
        items = self.query_handler.get_order_items(order_id)
        if order_result.get('order'):
            order_result['order']['items'] = items

        # Return as-is (already has 'order' key)
        return order_result

    def _query_stock(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query product stock."""
        product_name = entities.get("product_name")
        attributes = entities.get("attributes", {})
        brand = attributes.get("brand") if attributes else None
        color = attributes.get("color") if attributes else None
        size = attributes.get("size") if attributes else None

        if not product_name:
            # Return featured products if no specific product
            result = self.query_handler._get_featured_products(limit=5)
            if result.get('found') and result.get('products'):
                return {"products": result['products'], "type": "featured"}
            return {"error": "no_products", "message": "Tidak ada produk tersedia"}

        # Search with just product_name first (brand might already be in the name)
        result = self.query_handler.get_product_info(product_name)

        # If not found and brand exists but not already in product_name, try with brand
        if not result.get('found') and brand and brand.lower() not in product_name.lower():
            search_query = f"{product_name} {brand}"
            result = self.query_handler.get_product_info(search_query)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            product = result['products'][0]  # Get first matching product
            # Get stock information
            stock_result = self.query_handler.get_product_stock(product.get("id"))
            if stock_result.get('found'):
                product["stock"] = {"qty": stock_result.get('stock', 0)}
            else:
                product["stock"] = {"qty": product.get('stock', 0)}  # Use stock from product_flat
            product["attributes_requested"] = {"brand": brand, "color": color, "size": size}
            return {"product": product}
        else:
            return {
                "error": "product_not_found",
                "message": f"Produk '{product_name}' tidak ditemukan",
                "attributes": attributes
            }

    def _query_price(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query product price."""
        product_name = entities.get("product_name")
        attributes = entities.get("attributes", {})
        brand = attributes.get("brand") if attributes else None

        if not product_name:
            return {
                "error": "no_product",
                "message": "Produk mana yang ingin dicek harganya?"
            }

        # Search with just product_name first (brand might already be in the name)
        result = self.query_handler.get_product_info(product_name)

        # If not found and brand exists but not already in product_name, try with brand
        if not result.get('found') and brand and brand.lower() not in product_name.lower():
            search_query = f"{product_name} {brand}"
            result = self.query_handler.get_product_info(search_query)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            product = result['products'][0]  # Get first matching product
            product["attributes_requested"] = attributes
            return {"product": product}
        else:
            return {
                "error": "product_not_found",
                "message": f"Produk '{product_name}' tidak ditemukan"
            }

    def _query_payment(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query payment methods."""
        # Note: payment_method is no longer in new entity structure
        # but we keep the general payment query functionality
        result = self.query_handler.get_payment_methods()

        # Handle query_handler response format: {'found': True, 'methods': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('methods'):
            return {"found": True, "methods": result['methods']}
        return {"found": False, "message": "Metode pembayaran tidak tersedia"}

    def _query_product_info(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query detailed product information."""
        product_name = entities.get("product_name")
        attributes = entities.get("attributes", {})
        brand = attributes.get("brand") if attributes else None

        if not product_name:
            # Return featured products if no specific product
            result = self.query_handler._get_featured_products(limit=5)
            if result.get('found') and result.get('products'):
                return {"products": result['products'], "type": "featured"}
            return {
                "error": "no_product",
                "message": "Produk mana yang ingin diketahui informasinya?"
            }

        # Search with just product_name first (brand might already be in the name)
        result = self.query_handler.get_product_info(product_name)

        # If not found and brand exists but not already in product_name, try with brand
        if not result.get('found') and brand and brand.lower() not in product_name.lower():
            search_query = f"{product_name} {brand}"
            result = self.query_handler.get_product_info(search_query)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            product = result['products'][0]  # Get first matching product
            product["attributes_requested"] = attributes
            return {"product": product}
        else:
            return {
                "error": "product_not_found",
                "message": f"Produk '{product_name}' tidak ditemukan"
            }

    def _query_product_list(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query all available products (product_list task)."""
        result = self.query_handler._get_featured_products(limit=10)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            return {"products": result['products'], "type": "product_list"}
        return {
            "error": "no_products",
            "message": "Belum ada produk yang tersedia saat ini."
        }

    def _query_search_product(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Search products by keyword (search_product task)."""
        product_name = entities.get("product_name")
        attributes = entities.get("attributes", {})
        brand = attributes.get("brand") if attributes else None

        if not product_name:
            return {
                "error": "no_keyword",
                "message": "Kata kunci pencarian tidak ditemukan."
            }

        # Search with just product_name first (brand might already be in the name)
        result = self.query_handler.get_product_info(product_name)

        # If not found and brand exists but not already in product_name, try with brand
        if not result.get('found') and brand and brand.lower() not in product_name.lower():
            search_query = f"{product_name} {brand}"
            result = self.query_handler.get_product_info(search_query)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            product = result['products'][0]  # Get first matching product
            product["attributes_requested"] = attributes
            return {"product": product, "type": "search_result"}
        else:
            # Try to get similar products
            suggestions_result = self.query_handler._get_featured_products(limit=5)
            suggestions = suggestions_result.get('products', []) if suggestions_result.get('found') else []
            return {
                "error": "product_not_found",
                "message": f"Produk '{product_name}' tidak ditemukan",
                "suggestions": suggestions
            }

    def _query_product_by_category(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query products by category (product_by_category task)."""
        category = entities.get("category")

        if not category:
            return {
                "error": "no_category",
                "message": "Kategori apa yang ingin dilihat?"
            }

        # For now, return featured products with category info
        # TODO: Implement actual category filtering in query_handler
        result = self.query_handler._get_featured_products(limit=10)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            return {"products": result['products'], "type": "by_category", "category": category}
        return {
            "error": "no_products",
            "message": f"Belum ada produk dalam kategori '{category}'."
        }

    def _query_product_by_attribute(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Query products by attributes (product_by_attribute task)."""
        product_name = entities.get("product_name")
        attributes = entities.get("attributes", {})
        brand = attributes.get("brand") if attributes else None
        color = attributes.get("color") if attributes else None
        size = attributes.get("size") if attributes else None

        if not product_name and not brand:
            return {
                "error": "no_attributes",
                "message": "Produk atau atribut apa yang dicari?"
            }

        # Search with just product_name first (brand might already be in the name)
        if product_name:
            result = self.query_handler.get_product_info(product_name)

            # If not found and brand exists but not already in product_name, try with brand
            if not result.get('found') and brand and brand.lower() not in product_name.lower():
                search_query = f"{product_name} {brand}"
                result = self.query_handler.get_product_info(search_query)
        else:
            # Only brand provided
            result = self.query_handler.get_product_info(brand)

        # Handle query_handler response format: {'found': True, 'products': [...]}
        if result.get('error'):
            return result

        if result.get('found') and result.get('products'):
            product = result['products'][0]  # Get first matching product
            # Get stock information
            stock_result = self.query_handler.get_product_stock(product.get("id"))
            if stock_result.get('found'):
                product["stock"] = {"qty": stock_result.get('stock', 0)}
            else:
                product["stock"] = {"qty": product.get('stock', 0)}  # Use stock from product_flat
            product["attributes_requested"] = {"brand": brand, "color": color, "size": size}
            return {"product": product, "type": "by_attribute"}
        else:
            return {
                "error": "product_not_found",
                "message": f"Produk dengan kriteria tersebut tidak ditemukan",
                "attributes": attributes
            }

    def _generate_response(
        self,
        slots: SlotResult,
        db_result: Dict[str, Any]
    ) -> str:
        """
        Generate a response based on task and database results.

        Args:
            slots: Extracted slot information
            db_result: Database query results

        Returns:
            Response text in Indonesian
        """
        task = slots.task
        entities = slots.entities
        multi_intent = slots.multi_intent

        # Map LLM task to response generator intent
        intent = TASK_TYPES.get(task, task)

        # Handle errors
        if "error" in db_result:
            error_type = db_result.get("error")
            message = db_result.get("message", "Terjadi kesalahan")

            if error_type == "no_order_id":
                return "Mohon berikan nomor pesanan Anda agar saya bisa cek statusnya."
            elif error_type == "order_not_found":
                return message
            elif error_type == "product_not_found":
                return f"{message}. Silakan cek kembali nama produknya atau lihat katalog kami."
            elif error_type == "no_product":
                return message
            else:
                return f"Maaf, {message}"

        # Generate primary response
        response = self.response_generator.generate(
            intent=intent,
            entities=entities,
            db_result=db_result
        )

        # Add multi-intent information
        if multi_intent:
            response = self._add_multi_intent_info(response, multi_intent, entities, db_result)

        return response

    def _add_multi_intent_info(
        self,
        response: str,
        multi_intent: List[str],
        entities: Dict[str, Any],
        db_result: Dict[str, Any]
    ) -> str:
        """Add information for secondary intents."""
        for intent in multi_intent:
            if intent == "ask_price" and "product" in db_result:
                product = db_result["product"]
                price = product.get("price")
                if price:
                    response += f"\n\nHarga: Rp {price:,.0f}"

            elif intent == "check_stock" and "product" in db_result:
                product = db_result["product"]
                stock = product.get("stock", {})
                qty = stock.get("qty", 0) if stock else 0
                if qty > 0:
                    response += f"\n\nStok tersedia: {qty} unit"
                else:
                    response += "\n\nMaaf, stok sedang kosong."

        return response

    def _error_response(self, message: str, error: str) -> ChatResponse:
        """Create an error response."""
        return ChatResponse(
            response=message,
            task="error",
            entities={},
            confidence=0.0,
            multi_intent=[],
            error=error
        )


# Convenience function for quick testing
def create_chatbot(
    llm_server_url: str = "http://localhost:8080",
    db_config: Optional[Dict[str, Any]] = None,
    whatsapp_number: str = "+6281234567890"
) -> LLMChatbot:
    """Create a chatbot instance with default configuration."""
    from src.config import LLM_SERVER_URL, DB_CONFIG, WHATSAPP_NUMBER

    return LLMChatbot(
        llm_server_url=llm_server_url or LLM_SERVER_URL,
        db_config=db_config or DB_CONFIG,
        whatsapp_number=whatsapp_number or WHATSAPP_NUMBER
    )


if __name__ == "__main__":
    # Quick test
    from src.config import LLM_SERVER_URL, DB_CONFIG, WHATSAPP_NUMBER

    print("Initializing LLM Chatbot...")
    chatbot = LLMChatbot(
        llm_server_url=LLM_SERVER_URL,
        db_config=DB_CONFIG,
        whatsapp_number=WHATSAPP_NUMBER
    )

    if not chatbot.is_llm_available():
        print("ERROR: LLM server not available!")
        print("Start with: ./scripts/start_llm_server.sh")
        exit(1)

    print("LLM server is online!")
    print("-" * 50)

    test_queries = [
        # check_stock
        "stok hp xiaomi masih ada?",
        # ask_price
        "harga laptop asus berapa?",
        # check_order
        "pesanan 12345 udah sampai mana?",
        # ask_payment
        "bisa bayar pake gopay ga?",
        # product_list
        "produk ada apa aja?",
        # search_product
        "cari sepatu lari dong",
        # product_by_category
        "produk fashion ada apa aja?",
        # product_by_attribute
        "ada baju nike warna merah ukuran L?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = chatbot.process(query)
        print(f"Task: {result.task}")
        print(f"Entities: {result.entities}")
        print(f"Confidence: {result.confidence}")
        print(f"Response: {result.response}")
        print("-" * 50)
