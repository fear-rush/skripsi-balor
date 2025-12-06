"""
Conversation Context Manager for Multi-Turn Conversations

Manages conversation state, history, and context resolution for
the LLM-based e-commerce chatbot.

Key Features:
    - Track conversation history
    - Handle pending clarifications (e.g., waiting for order_id)
    - Maintain active entities across turns
    - Resolve pronouns and references

Usage:
    from src.conversation_context import ConversationContext, ContextStore

    store = ContextStore()
    context = store.get_or_create("session_123")
    context.add_turn(user_query, bot_response, slots)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    timestamp: datetime
    user_query: str
    bot_response: str
    task: str
    entities: Dict[str, Any]
    confidence: float
    needs_clarification: bool = False
    clarification_slot: Optional[str] = None  # Which slot we're waiting for


@dataclass
class PendingClarification:
    """Tracks a pending clarification request."""
    slot_name: str  # e.g., "order_id", "product_name"
    task: str  # e.g., "check_order", "check_stock"
    original_query: str  # The query that triggered the clarification
    timestamp: datetime
    entities: Dict[str, Any]  # Any entities already extracted


class ConversationContext:
    """
    Manages conversation state and history for a single session.

    Attributes:
        session_id: Unique session identifier
        history: List of conversation turns
        pending_clarification: Current pending clarification (if any)
        active_entities: Entities tracked across turns
        created_at: Session creation time
        updated_at: Last activity time
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[ConversationTurn] = []
        self.pending_clarification: Optional[PendingClarification] = None
        self.active_entities: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_turn(
        self,
        user_query: str,
        bot_response: str,
        task: str,
        entities: Dict[str, Any],
        confidence: float,
        needs_clarification: bool = False,
        clarification_slot: Optional[str] = None
    ):
        """
        Add a conversation turn to history.

        Args:
            user_query: User's input
            bot_response: Bot's response
            task: Identified task type
            entities: Extracted entities
            confidence: Confidence score
            needs_clarification: Whether bot asked for clarification
            clarification_slot: Which slot needs clarification
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            bot_response=bot_response,
            task=task,
            entities=entities,
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification_slot=clarification_slot
        )
        self.history.append(turn)
        self.updated_at = datetime.now()

        # Update active entities
        self._update_active_entities(entities)

        # Handle clarification state
        if needs_clarification and clarification_slot:
            self.set_pending_clarification(
                slot_name=clarification_slot,
                task=task,
                original_query=user_query,
                entities=entities
            )
        else:
            # Clear pending clarification if this turn resolved it
            self.clear_pending_clarification()

        logger.debug(f"Added turn for session {self.session_id}: task={task}")

    def _update_active_entities(self, entities: Dict[str, Any]):
        """Update active entities with new values, keeping old ones if not overwritten."""
        if not entities:
            return

        for key, value in entities.items():
            if value is not None and value != "" and value != {}:
                if key == "attributes" and isinstance(value, dict):
                    # Merge attributes
                    if "attributes" not in self.active_entities:
                        self.active_entities["attributes"] = {}
                    for attr_key, attr_value in value.items():
                        if attr_value is not None:
                            self.active_entities["attributes"][attr_key] = attr_value
                else:
                    self.active_entities[key] = value

    def set_pending_clarification(
        self,
        slot_name: str,
        task: str,
        original_query: str,
        entities: Dict[str, Any]
    ):
        """Set a pending clarification request."""
        self.pending_clarification = PendingClarification(
            slot_name=slot_name,
            task=task,
            original_query=original_query,
            timestamp=datetime.now(),
            entities=entities.copy() if entities else {}
        )
        logger.info(f"Session {self.session_id}: Waiting for {slot_name} (task: {task})")

    def clear_pending_clarification(self):
        """Clear the pending clarification state."""
        if self.pending_clarification:
            logger.info(f"Session {self.session_id}: Cleared pending clarification")
        self.pending_clarification = None

    def has_pending_clarification(self) -> bool:
        """Check if there's a pending clarification."""
        return self.pending_clarification is not None

    def get_pending_slot(self) -> Optional[str]:
        """Get the slot name we're waiting for."""
        if self.pending_clarification:
            return self.pending_clarification.slot_name
        return None

    def get_pending_task(self) -> Optional[str]:
        """Get the task associated with pending clarification."""
        if self.pending_clarification:
            return self.pending_clarification.task
        return None

    def get_pending_entities(self) -> Dict[str, Any]:
        """Get entities from the pending clarification."""
        if self.pending_clarification:
            return self.pending_clarification.entities
        return {}

    def resolve_clarification_input(self, user_input: str) -> Dict[str, Any]:
        """
        Resolve user input as a clarification response.

        Args:
            user_input: User's response to clarification request

        Returns:
            Dict with resolved task, entities, and whether this resolves the clarification
        """
        if not self.pending_clarification:
            return {"resolved": False}

        slot_name = self.pending_clarification.slot_name
        task = self.pending_clarification.task
        entities = self.pending_clarification.entities.copy()

        # Clean the input
        clean_input = user_input.strip()

        # Handle order_id clarification
        if slot_name == "order_id":
            # Extract numeric order ID or use as-is
            order_id = self._extract_order_id(clean_input)
            if order_id:
                entities["order_id"] = order_id
                return {
                    "resolved": True,
                    "task": task,
                    "entities": entities,
                    "slot_name": slot_name,
                    "slot_value": order_id
                }

        # Handle product_name clarification
        elif slot_name == "product_name":
            if len(clean_input) > 0:
                entities["product_name"] = clean_input
                return {
                    "resolved": True,
                    "task": task,
                    "entities": entities,
                    "slot_name": slot_name,
                    "slot_value": clean_input
                }

        # Handle category clarification
        elif slot_name == "category":
            if len(clean_input) > 0:
                entities["category"] = clean_input
                return {
                    "resolved": True,
                    "task": task,
                    "entities": entities,
                    "slot_name": slot_name,
                    "slot_value": clean_input
                }

        # Generic handling for other slots
        elif len(clean_input) > 0:
            entities[slot_name] = clean_input
            return {
                "resolved": True,
                "task": task,
                "entities": entities,
                "slot_name": slot_name,
                "slot_value": clean_input
            }

        return {"resolved": False}

    def _extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from text (handles various formats)."""
        import re

        # Remove common prefixes
        text = text.lower().strip()
        text = text.replace('#', '').replace('order', '').replace('pesanan', '')
        text = text.replace('no', '').replace('nomor', '').strip()

        # Find numeric pattern
        match = re.search(r'\d+', text)
        if match:
            return match.group()

        # If text itself is alphanumeric order ID
        clean = re.sub(r'[^a-zA-Z0-9]', '', text)
        if clean:
            return clean

        return None

    def get_last_product(self) -> Optional[str]:
        """Get the last mentioned product name."""
        return self.active_entities.get("product_name")

    def get_last_task(self) -> Optional[str]:
        """Get the task from the last turn."""
        if self.history:
            return self.history[-1].task
        return None

    def get_conversation_summary(self) -> str:
        """Get a brief summary of the conversation."""
        if not self.history:
            return "New conversation"

        turns = len(self.history)
        last_task = self.history[-1].task if self.history else "none"
        pending = f", waiting for {self.pending_clarification.slot_name}" if self.pending_clarification else ""

        return f"{turns} turns, last task: {last_task}{pending}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "turn_count": len(self.history),
            "active_entities": self.active_entities,
            "has_pending_clarification": self.has_pending_clarification(),
            "pending_slot": self.get_pending_slot(),
            "last_task": self.get_last_task()
        }


class ContextStore:
    """
    In-memory storage for conversation contexts.

    Manages multiple sessions with automatic cleanup of expired sessions.
    """

    def __init__(self, max_sessions: int = 1000, ttl_minutes: int = 30):
        """
        Initialize context store.

        Args:
            max_sessions: Maximum number of sessions to keep
            ttl_minutes: Time-to-live for inactive sessions
        """
        self.sessions: Dict[str, ConversationContext] = {}
        self.max_sessions = max_sessions
        self.ttl = timedelta(minutes=ttl_minutes)

    def get_or_create(self, session_id: str) -> ConversationContext:
        """
        Get existing context or create new one.

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationContext for the session
        """
        # Cleanup expired sessions periodically
        if len(self.sessions) > self.max_sessions * 0.8:
            self.cleanup_expired()

        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(session_id)
            logger.info(f"Created new session: {session_id}")
        else:
            # Update last access time
            self.sessions[session_id].updated_at = datetime.now()

        return self.sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationContext]:
        """Get context if it exists."""
        return self.sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session context."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def cleanup_expired(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            session_id for session_id, context in self.sessions.items()
            if now - context.updated_at > self.ttl
        ]

        for session_id in expired:
            del self.sessions[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)

    def get_all_session_ids(self) -> List[str]:
        """Get all active session IDs."""
        return list(self.sessions.keys())


# Global context store instance
_context_store: Optional[ContextStore] = None


def get_context_store() -> ContextStore:
    """Get or create the global context store."""
    global _context_store
    if _context_store is None:
        _context_store = ContextStore()
    return _context_store


# Mapping of task to clarification slot
CLARIFICATION_SLOT_MAP = {
    "check_order": "order_id",
    "check_stock": "product_name",
    "ask_price": "product_name",
    "product_info": "product_name",
    "search_product": "product_name",
    "product_by_category": "category",
    "product_by_attribute": "product_name",
}


def determine_clarification_slot(task: str, entities: Dict[str, Any]) -> Optional[str]:
    """
    Determine which slot needs clarification based on task and entities.

    Args:
        task: The identified task
        entities: Currently extracted entities

    Returns:
        Slot name that needs clarification, or None
    """
    if task == "check_order":
        if not entities.get("order_id"):
            return "order_id"

    elif task in ["check_stock", "ask_price", "product_info", "search_product"]:
        if not entities.get("product_name"):
            return "product_name"

    elif task == "product_by_category":
        if not entities.get("category"):
            return "category"

    elif task == "product_by_attribute":
        if not entities.get("product_name") and not entities.get("attributes", {}).get("brand"):
            return "product_name"

    return None
