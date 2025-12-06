# LLM Slot Extractor Integration Plan

**Date**: 2025-12-06
**Status**: Implementation Plan
**Goal**: Integrate Qwen3-4B-Instruct-2507 into backend and web UI

---

## Executive Summary

This document outlines the plan to integrate the LLM-based slot extractor (Qwen3-4B-Instruct-2507) into the e-commerce chatbot backend and web UI, replacing the V4 intent classifier approach.

### Current State
- Web app uses `chatbot.py` (v1) with 3-intent classifier
- `chatbot_v2.py` exists but uses V4 classifier (6 intents)
- `llm_slot_extractor.py` is ready but not integrated
- Multiple legacy files causing confusion

### Target State
- Web app uses LLM slot extractor (Qwen3-4B-2507)
- Single clean chatbot module
- Response generator updated for LLM output
- Removed/archived legacy files

---

## Codebase Analysis

### Files to DELETE

| File | Lines | Reason |
|------|-------|--------|
| `src/chatbot.py` | 156 | Legacy v1, replaced by LLM |
| `src/response_generator.py` | 149 | Legacy v1 templates |
| `src/llm_query_generator.py` | 514 | Text-to-SQL not needed |
| `main.py` | 4 | Orphaned, never used |
| `test.py` | 100+ | Move to scripts/ |

### Files to ARCHIVE (models/archive/)

| File/Folder | Size | Reason |
|-------------|------|--------|
| `models/intent_classifier/` | ~270MB | V1 classifier, superseded |
| `models/intent_classifier_v4/` | ~270MB | V4 classifier, replaced by LLM |
| `models/llm/Qwen3-8B-Q4_K_M.gguf` | ~5GB | Caused 42.5% error rate |
| `models/llm/qwen2.5-3b-instruct-q8_0.gguf` | ~3.6GB | Superseded by Qwen3-4B |
| `src/entity_extractor.py` | 425 lines | LLM does entity extraction |

### Files to KEEP & REFACTOR

| File | Action |
|------|--------|
| `src/llm_slot_extractor.py` | Keep, minor updates |
| `src/query_handler.py` | Refactor for LLM entity format |
| `src/response_generator_v2.py` | Rename, add multi-intent support |
| `src/chatbot_v2.py` | Replace entirely with LLM version |

---

## New Architecture

### File Structure

```
src/
├── __init__.py
├── chatbot.py              # NEW: LLM-based chatbot
├── llm_slot_extractor.py   # KEEP: Core LLM extraction
├── query_handler.py        # KEEP: Database queries (refactored)
├── response_generator.py   # RENAMED from v2
└── config.py               # NEW: Centralized configuration

web/
├── app.py                  # UPDATED: Uses LLM chatbot
├── templates/
│   └── index.html          # UPDATED: LLM status indicator
└── static/

models/
├── llm/
│   └── Qwen3-4B-Instruct-2507-Q4_K_M.gguf
└── archive/
    ├── intent_classifier/
    └── intent_classifier_v4/
```

### Processing Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────┐
│         LLMChatbot (chatbot.py)         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      SlotExtractor (llm_slot_extractor) │
│  Returns: task, entities, multi_intent  │
│           confidence, clarification     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│     BagistoQueryHandler (query_handler) │
│  Uses extracted entities for DB queries │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│    ResponseGenerator (response_generator)│
│  Generates response + multi-intent info │
└─────────────────────────────────────────┘
    ↓
Response + Metadata (JSON)
```

---

## Key Changes

### 1. New Chatbot Module (`src/chatbot.py`)

```python
class LLMChatbot:
    def __init__(self, llm_server_url, db_config, whatsapp_number):
        self.slot_extractor = SlotExtractor(server_url=llm_server_url)
        self.query_handler = BagistoQueryHandler(**db_config)
        self.response_generator = ResponseGenerator(whatsapp_number)

    def process(self, query: str) -> ChatResponse:
        # 1. Extract slots using LLM
        slots = self.slot_extractor.extract(query)

        # 2. Handle clarification if needed
        if slots.needs_clarification:
            return self._handle_clarification(slots)

        # 3. Query database
        db_result = self._query_database(slots)

        # 4. Generate response
        response = self.response_generator.generate(
            task=slots.task,
            entities=slots.entities,
            db_result=db_result,
            multi_intent=slots.multi_intent
        )

        return ChatResponse(...)
```

### 2. Configuration Module (`src/config.py`)

```python
import os

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8080")
LLM_MODEL = "Qwen3-4B-Instruct-2507"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "bagisto_db"),
    "port": int(os.getenv("DB_PORT", 3306))
}

WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+6281234567890")
CONFIDENCE_THRESHOLD = 0.7
```

### 3. Updated Query Handler

- Accept LLM entities dict instead of individual parameters
- Use `entities['product_name']` and `entities['variant']`
- Use `entities['order_id']` directly
- Filter by `entities['payment_method']`

### 4. Updated Response Generator

- Rename from `response_generator_v2.py`
- Add multi-intent response support
- Handle variant entity in product responses

### 5. Enhanced Web UI

- Detailed LLM status indicator (online/offline)
- Show latency and model info
- Display extracted entities
- Show multi-intent when detected
- Poll `/llm-health` endpoint every 10 seconds

---

## New API Format

### Chat Endpoint Response

```json
{
  "response": "Stok sepatu nike size 42 tersedia. Harga Rp 1.500.000",
  "task": "check_stock",
  "entities": {
    "product_name": "sepatu nike",
    "variant": "size 42",
    "order_id": null,
    "payment_method": null,
    "quantity": null
  },
  "confidence": 0.9,
  "multi_intent": ["ask_price"]
}
```

### LLM Health Endpoint

```json
{
  "status": "online",
  "latency_ms": 45.2,
  "model": "Qwen3-4B-Instruct-2507",
  "server_url": "http://localhost:8080"
}
```

---

## Benefits of LLM Approach

| Aspect | V4 Classifier | LLM Slot Extractor |
|--------|---------------|-------------------|
| Accuracy | 78.75% | **96.25%** |
| Entity Extraction | No | **Yes** |
| Multi-Intent | No | **Yes** |
| Clarification | No | **Yes** |
| Zero-shot | No | **Yes** |
| Latency | 41ms | 2260ms |

---

## Implementation Steps

1. **Cleanup**: Create archive, move deprecated files
2. **Config**: Create `src/config.py`
3. **Chatbot**: Create new `src/chatbot.py` with LLMChatbot
4. **Query Handler**: Update for LLM entity format
5. **Response Generator**: Add multi-intent support
6. **Web App**: Update imports, add health endpoint
7. **UI**: Add LLM status indicator
8. **Documentation**: Update README

---

## Success Criteria

1. LLM server running with Qwen3-4B-Instruct-2507
2. Web app sends query → LLM extracts slots → DB query → Response
3. Entities displayed in UI
4. Multi-intent detected and shown
5. Confidence score displayed
6. LLM status indicator working
7. All legacy files cleaned up

---

*Generated: 2025-12-06*
