# E-Commerce Chatbot Improvement Implementation Plan

## Overview

This plan outlines a comprehensive improvement of the Bagisto e-commerce chatbot for a thesis project. The goal is to transform a basic intent classifier with 71% real-world accuracy into a robust NLU pipeline with:
- **6 intents** (expanded from 3)
- **Semantic entity extraction** (replacing hardcoded keywords)
- **LLM-based Text-to-SQL** for dynamic product queries
- **Dual fallback mechanism** for out-of-scope queries

The improvements will be documented with before/after comparisons, ablation studies, and visualizations suitable for thesis presentation.

---

## Current State Analysis

### What Exists Now:

| Component | Location | Description |
|-----------|----------|-------------|
| Intent Classifier | `models/intent_classifier/best_model/` | DistilBERT fine-tuned on synthetic data |
| Training Notebook | `notebook/skripsi_chatbot.ipynb` | Data generation + training pipeline |
| Chatbot Backend | `src/chatbot.py` | Main pipeline orchestrator |
| Query Handler | `src/query_handler.py` | MySQL queries for Bagisto DB |
| Response Generator | `src/response_generator.py` | Template-based responses |

### Current Performance:

| Metric | Synthetic Test | Real-World | Gap |
|--------|---------------|------------|-----|
| Accuracy | 99.11% | 71.43% | **27.68%** |
| Correct | 223/225 | 15/21 | - |

### Key Problems Identified:

1. **Overfitting to Synthetic Data**
   - Training data too "clean" (only 15% typo rate)
   - No out-of-scope examples
   - No multi-intent queries
   - File: `notebook/skripsi_chatbot.ipynb` cells 1-3

2. **Hardcoded Entity Extraction**
   - Only 7 product keywords recognized
   - File: `src/chatbot.py:56-60`
   ```python
   keywords = ['sepatu', 'laptop', 'kaos', 'tas', 'baju', 'celana', 'hp']
   ```

3. **No Fallback Mechanism**
   - Model can't say "I don't understand"
   - Returns all products when query unclear
   - No confidence thresholding

4. **Limited Intent Granularity**
   - `product_info` handles price, stock, and description together
   - Can't distinguish "berapa harga X?" from "stok X ada?"

---

## Desired End State

After implementation, the chatbot will:

1. **Classify 6 intents** with >85% real-world accuracy:
   - `order_status` - "cek pesanan #123"
   - `payment_info` - "bisa bayar pakai apa?"
   - `product_price` - "harga sepatu berapa?"
   - `product_stock` - "stok laptop ada?"
   - `product_description` - "spesifikasi HP gimana?"
   - `out_of_scope` - "mau refund", "jam buka toko"

2. **Extract entities dynamically** using semantic matching against Bagisto product database

3. **Generate SQL queries** via LLM for complex product queries

4. **Handle edge cases gracefully**:
   - Low confidence → redirect to WhatsApp admin
   - Out-of-scope → polite fallback message
   - Extreme typos → still classify correctly

### Verification Criteria:
- [ ] Real-world accuracy >85% (vs current 71%)
- [ ] Fallback triggers correctly for out-of-scope queries
- [ ] Product extraction works for ALL Bagisto products (not just 7 keywords)
- [ ] Thesis includes before/after comparison charts

---

## What We're NOT Doing

To prevent scope creep, the following are explicitly out of scope:

1. **Multi-language support** - Indonesian only (no English)
2. **Voice/speech input** - Text only
3. **Multi-turn conversation** - Single query/response (no context memory)
4. **Real-time training** - Offline training only
5. **Deployment to production** - Local development only
6. **Mobile app integration** - Web chatbot template only
7. **Admin dashboard** - Using existing Bagisto admin

---

## Implementation Approach

The implementation follows a **layered improvement strategy**:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Intent Classifier v2 (6 intents + confidence)     │
│  - Trained on improved synthetic data (Phase 1)             │
│  - Returns intent + confidence score                        │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     confidence >= 0.7           confidence < 0.7
              │                           │
              ▼                           ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ Phase 3: Entity Extract │    │ Fallback to WhatsApp    │
│ - Semantic matching     │    │ Admin Contact           │
│ - Product name from DB  │    └─────────────────────────┘
└─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: LLM Text-to-SQL (for product queries)             │
│  - Schema-aware query generation                            │
│  - Fallback to direct DB query if LLM fails                 │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Response Generator                                          │
│  - Template-based for order/payment                         │
│  - Dynamic for product queries                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Baseline & Data Generation Overhaul

### Overview
Establish baseline metrics for current model, then create improved synthetic data generator with 6 intents, more realistic variations, and edge cases.

### Changes Required:

#### 1.1 Create Baseline Evaluation Script
**File**: `notebook/01_baseline_evaluation.ipynb`
**Purpose**: Document current model performance before any changes

```python
# Cell 1: Load current model and run comprehensive evaluation
from transformers import pipeline
import pandas as pd
import json

# Load current model
classifier = pipeline(
    "text-classification",
    model="./models/intent_classifier/best_model",
    device=-1
)

# Extended real-world test cases (50+ queries)
BASELINE_TEST_CASES = [
    # Order status - easy
    {"text": "cek pesanan 12345", "expected": "order_status", "category": "easy"},
    {"text": "order saya udah sampai mana", "expected": "order_status", "category": "easy"},

    # Order status - typos
    {"text": "ordr sy mn", "expected": "order_status", "category": "typo"},
    {"text": "psnan gw kmn bro", "expected": "order_status", "category": "typo"},

    # Order status - verbose
    {"text": "halo min mau tanya dong pesanan saya yang kemarin itu udah dikirim belum ya", "expected": "order_status", "category": "verbose"},

    # Payment - easy
    {"text": "bisa bayar pakai gopay?", "expected": "payment_info", "category": "easy"},
    {"text": "metode pembayaran apa aja", "expected": "payment_info", "category": "easy"},

    # Payment - typos
    {"text": "byr pk ap", "expected": "payment_info", "category": "typo"},

    # Product queries (will be split in v2)
    {"text": "harga laptop berapa", "expected": "product_info", "category": "easy"},
    {"text": "stok sepatu ada ga", "expected": "product_info", "category": "easy"},
    {"text": "spesifikasi hp gimana", "expected": "product_info", "category": "easy"},

    # Ambiguous/multi-intent
    {"text": "mau beli sepatu bisa cod ga", "expected": "payment_info", "category": "ambiguous"},
    {"text": "harga berapa kalau bayar cash", "expected": "product_info", "category": "ambiguous"},

    # Out of scope (current model has no handling)
    {"text": "mau refund dong", "expected": None, "category": "out_of_scope"},
    {"text": "barang rusak mau komplain", "expected": None, "category": "out_of_scope"},
    {"text": "jam buka toko kapan", "expected": None, "category": "out_of_scope"},
    {"text": "alamat warehouse dimana", "expected": None, "category": "out_of_scope"},
    {"text": "cancel pesanan", "expected": None, "category": "out_of_scope"},

    # Add 30+ more test cases covering all categories...
]

# Run evaluation and save baseline metrics
results = []
for test in BASELINE_TEST_CASES:
    pred = classifier(test["text"])[0]
    results.append({
        "text": test["text"],
        "expected": test["expected"],
        "predicted": pred["label"],
        "confidence": pred["score"],
        "category": test["category"],
        "correct": pred["label"] == test["expected"] if test["expected"] else None
    })

baseline_df = pd.DataFrame(results)
baseline_df.to_csv("./evaluation/baseline_v1_results.csv", index=False)

# Calculate metrics by category
print("=== BASELINE V1 METRICS ===")
for category in baseline_df["category"].unique():
    cat_df = baseline_df[baseline_df["category"] == category]
    if cat_df["correct"].notna().any():
        acc = cat_df["correct"].mean()
        print(f"{category}: {acc:.2%}")
```

#### 1.2 Improved Synthetic Data Generator
**File**: `notebook/02_improved_data_generator.ipynb`
**Purpose**: Generate 6-intent dataset with realistic variations

```python
# Cell 1: Intent patterns for 6 intents

import json
import random
import pandas as pd
from typing import List, Dict

# ============================================================
# 6 INTENT PATTERNS (Indonesian)
# ============================================================

ORDER_STATUS_PATTERNS = [
    # Base patterns
    "cek pesanan {order_id}",
    "status order {order_id}",
    "pesanan {order_id} udah sampai mana",
    "tracking pesanan {order_id}",
    "orderan saya gimana",
    "mau cek status pesanan",
    "barang saya udah dikirim belum",
    "order {order_id} udah diproses belum",
    "kapan pesanan {order_id} sampai",
    "estimasi pengiriman order {order_id}",

    # Casual/slang
    "order gue {order_id} mana",
    "pesanan gw {order_id} kemana",
    "brg {order_id} dmn",
    "udah dikirim blm {order_id}",
    "kok lama bgt order {order_id}",

    # Verbose
    "halo min mau tanya dong pesanan saya nomor {order_id} itu udah sampai mana ya",
    "permisi kak orderan saya yang {order_id} kok belum dateng",
    "maaf ganggu mau cek status order {order_id} dong",
]

PAYMENT_INFO_PATTERNS = [
    # Base patterns
    "metode pembayaran apa aja",
    "bisa bayar pakai {payment_method}",
    "cara bayar gimana",
    "accept {payment_method} ga",
    "pembayaran via apa",
    "terima {payment_method} tidak",
    "ada cicilan ga",
    "bisa transfer bank",
    "support ewallet ga",

    # Casual/slang
    "bayar pake apa sih",
    "bs {payment_method} g",
    "pake {payment_method} bs?",
    "metode byr apa",

    # Verbose
    "mau tanya untuk pembayarannya bisa pakai metode apa aja ya",
    "kalo mau bayar pake {payment_method} bisa ga min",
]

PRODUCT_PRICE_PATTERNS = [
    # Base patterns
    "harga {product} berapa",
    "berapa harga {product}",
    "{product} harganya berapa",
    "price {product}",
    "{product} berapa duit",
    "kisaran harga {product}",
    "range harga {product}",

    # Casual/slang
    "hrg {product} brp",
    "{product} brp duit",
    "brp {product}",
    "{product} mahal ga",

    # Verbose
    "mau tanya dong harga {product} yang ada di toko berapa ya",
    "kira kira {product} itu harganya berapa ya kak",
]

PRODUCT_STOCK_PATTERNS = [
    # Base patterns
    "stok {product} ada",
    "{product} ready stock",
    "{product} masih ada",
    "tersedia {product} ga",
    "available {product}",
    "{product} kosong ga",
    "sisa stok {product}",
    "stock {product} berapa",

    # Casual/slang
    "ad stok {product} g",
    "{product} ready?",
    "{product} msh ad?",
    "stok {product} abis blm",

    # Verbose
    "mau tanya apakah {product} masih tersedia stoknya",
    "kak {product} masih ada ga stoknya",
]

PRODUCT_DESCRIPTION_PATTERNS = [
    # Base patterns
    "info produk {product}",
    "detail {product}",
    "spesifikasi {product}",
    "deskripsi {product}",
    "{product} terbuat dari apa",
    "bahan {product} apa",
    "fitur {product}",
    "kelebihan {product}",
    "{product} ukurannya apa aja",
    "warna {product} ada apa aja",

    # Casual/slang
    "{product} gmn sih",
    "spec {product}",
    "{product} bagus ga",

    # Verbose
    "bisa dijelasin tentang produk {product} dong",
    "mau tau lebih detail tentang {product}",
]

OUT_OF_SCOPE_PATTERNS = [
    # Refund/return
    "mau refund",
    "cara refund gimana",
    "bisa return barang ga",
    "mau kembalikan barang",
    "uang saya kapan dikembalikan",
    "proses refund berapa lama",

    # Complaints
    "barang rusak",
    "mau komplain",
    "produk tidak sesuai",
    "kecewa sama pelayanan",
    "barang cacat",

    # Store info
    "jam buka toko",
    "alamat toko dimana",
    "lokasi warehouse",
    "bisa ambil langsung ga",
    "ada toko offline",

    # Order modification
    "cancel pesanan",
    "ubah alamat pengiriman",
    "ganti ukuran",
    "tambah item ke pesanan",

    # Other
    "mau kerja disini",
    "lowongan kerja ada ga",
    "jadi reseller gimana",
    "halo",
    "test",
    "siapa kamu",
]

# ============================================================
# ENTITIES
# ============================================================

ORDER_IDS = [
    "12345", "67890", "1", "99", "123",
    "#12345", "ORD-001", "nomor 555",
    "id 7788", "order-123", "pesanan 999"
]

PAYMENT_METHODS = [
    "gopay", "ovo", "dana", "shopeepay", "linkaja",
    "transfer bank", "bca", "mandiri", "bni", "bri",
    "cod", "cash", "credit card", "kartu kredit", "debit",
    "qris", "virtual account", "va"
]

# ============================================================
# DYNAMIC PRODUCT FETCHING FROM BAGISTO DATABASE
# ============================================================

import mysql.connector

def fetch_products_from_bagisto(db_config: dict = None) -> List[str]:
    """
    Dynamically fetch product names from Bagisto database.
    This ensures training data always reflects current inventory.

    Args:
        db_config: Database configuration dict

    Returns:
        List of product names and keywords for synthetic data
    """
    if db_config is None:
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'bagisto_db',
            'port': 3306
        }

    products = []

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Fetch all active product names
        query = """
            SELECT DISTINCT
                pf.name,
                pf.short_description
            FROM product_flat pf
            WHERE pf.status = 1
              AND pf.visible_individually = 1
              AND pf.name IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        for row in results:
            name = row['name']
            products.append(name)

            # Also add individual words from product name as keywords
            # e.g., "Arctic Cozy Knit Beanie" -> ["arctic", "cozy", "knit", "beanie"]
            words = name.lower().split()
            for word in words:
                if len(word) > 3 and word not in products:
                    products.append(word)

            # Add keywords from description
            desc = row.get('short_description', '') or ''
            desc_words = desc.lower().split()[:10]  # First 10 words
            for word in desc_words:
                if len(word) > 4 and word.isalpha() and word not in products:
                    products.append(word)

        cursor.close()
        conn.close()

        print(f"Fetched {len(products)} product keywords from Bagisto database")

    except Exception as e:
        print(f"Warning: Could not fetch from database: {e}")
        print("Using fallback generic product terms...")
        # Fallback to generic Indonesian product terms (NOT hardcoded products)
        products = [
            "produk", "barang", "item",
            "topi", "baju", "celana", "sepatu", "tas",
            "aksesoris", "pakaian", "fashion"
        ]

    return list(set(products))  # Remove duplicates

# Fetch products dynamically
PRODUCTS = fetch_products_from_bagisto()
print(f"Loaded {len(PRODUCTS)} product keywords for training data")

# ============================================================
# NOISE FUNCTIONS (MORE AGGRESSIVE)
# ============================================================

INDONESIAN_TYPO_MAP = {
    'a': ['4', '@', ''],
    'e': ['3', ''],
    'i': ['1', '!', ''],
    'o': ['0', ''],
    's': ['$', '5'],
    'g': ['6', '9'],
}

def introduce_typo(text: str, prob: float = 0.3) -> str:
    """Add typos with higher probability"""
    if random.random() > prob or len(text) < 5:
        return text

    words = text.split()
    if not words:
        return text

    # Apply typo to 1-2 words
    num_typos = random.randint(1, min(2, len(words)))

    for _ in range(num_typos):
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]

        if len(word) > 3:
            typo_type = random.choice(['remove', 'swap', 'replace'])

            if typo_type == 'remove':
                # Remove random character
                char_idx = random.randint(1, len(word) - 2)
                word = word[:char_idx] + word[char_idx+1:]

            elif typo_type == 'swap':
                # Swap adjacent characters
                if len(word) > 2:
                    char_idx = random.randint(0, len(word) - 2)
                    word = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]

            elif typo_type == 'replace':
                # Replace with similar character
                char_idx = random.randint(0, len(word) - 1)
                char = word[char_idx].lower()
                if char in INDONESIAN_TYPO_MAP:
                    replacement = random.choice(INDONESIAN_TYPO_MAP[char])
                    word = word[:char_idx] + replacement + word[char_idx+1:]

        words[word_idx] = word

    return " ".join(words)

def apply_abbreviation(text: str, prob: float = 0.25) -> str:
    """Apply common Indonesian abbreviations"""
    if random.random() > prob:
        return text

    abbreviations = {
        'yang': 'yg',
        'dengan': 'dgn',
        'sudah': 'udh',
        'belum': 'blm',
        'tidak': 'ga',
        'bisa': 'bs',
        'bagaimana': 'gmn',
        'gimana': 'gmn',
        'dimana': 'dmn',
        'kemana': 'kmn',
        'kapan': 'kpn',
        'kenapa': 'knp',
        'berapa': 'brp',
        'harga': 'hrg',
        'barang': 'brg',
        'pesanan': 'psnan',
        'tolong': 'tlg',
        'terima kasih': 'makasih',
        'makasih': 'thx',
        'saya': 'sy',
        'aku': 'ak',
        'kamu': 'km',
        'ada': 'ad',
        'atau': 'ato',
    }

    text_lower = text.lower()
    for full, abbrev in abbreviations.items():
        if full in text_lower and random.random() < 0.5:
            text = text.replace(full, abbrev)
            text = text.replace(full.capitalize(), abbrev)

    return text

PREFIXES = ["", "halo", "hi", "hai", "permisi", "min", "kak", "gan", "misi", "bang", "sis", "admin"]
SUFFIXES = ["", "dong", "ya", "pls", "please", "tolong", "thanks", "thx", "makasih", "?", "??"]

def wrap_casual(text: str, prob: float = 0.4) -> str:
    """Add casual prefix/suffix"""
    if random.random() > prob:
        return text

    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)

    if prefix:
        text = f"{prefix} {text}"
    if suffix and not text.endswith('?'):
        text = f"{text} {suffix}"

    return text.strip()

def add_noise(text: str, noise_level: str = "medium") -> str:
    """Apply noise based on level"""

    # Noise probabilities by level
    noise_config = {
        "low": {"typo": 0.1, "abbrev": 0.1, "casual": 0.2},
        "medium": {"typo": 0.25, "abbrev": 0.25, "casual": 0.35},
        "high": {"typo": 0.4, "abbrev": 0.4, "casual": 0.5},
        "extreme": {"typo": 0.6, "abbrev": 0.6, "casual": 0.7},
    }

    config = noise_config.get(noise_level, noise_config["medium"])

    # Apply transformations
    text = apply_abbreviation(text, config["abbrev"])
    text = introduce_typo(text, config["typo"])
    text = wrap_casual(text, config["casual"])

    # Random case variation
    case_choice = random.choice(["lower", "original", "capitalize"])
    if case_choice == "lower":
        text = text.lower()
    elif case_choice == "capitalize":
        text = text.capitalize()

    return text

# ============================================================
# GENERATOR
# ============================================================

def generate_intent_data(
    patterns: List[str],
    intent_label: str,
    entities: Dict[str, List[str]],
    num_samples: int = 500
) -> List[Dict]:
    """Generate samples for a single intent"""

    data = []
    noise_levels = ["low", "medium", "high", "extreme"]
    noise_weights = [0.3, 0.4, 0.2, 0.1]  # Most samples are medium noise

    for _ in range(num_samples):
        pattern = random.choice(patterns)

        # Replace placeholders
        if "{order_id}" in pattern:
            pattern = pattern.replace("{order_id}", random.choice(entities.get("order_ids", ["12345"])))
        if "{payment_method}" in pattern:
            pattern = pattern.replace("{payment_method}", random.choice(entities.get("payment_methods", ["gopay"])))
        if "{product}" in pattern:
            pattern = pattern.replace("{product}", random.choice(entities.get("products", ["sepatu"])))

        # Apply noise
        noise_level = random.choices(noise_levels, weights=noise_weights)[0]
        text = add_noise(pattern, noise_level)

        data.append({
            "text": text,
            "intent": intent_label,
            "noise_level": noise_level
        })

    return data

# ============================================================
# GENERATE FULL DATASET
# ============================================================

print("Generating 6-intent synthetic dataset...")

entities = {
    "order_ids": ORDER_IDS,
    "payment_methods": PAYMENT_METHODS,
    "products": PRODUCTS
}

dataset = []

# Generate 500 samples per intent
dataset.extend(generate_intent_data(ORDER_STATUS_PATTERNS, "order_status", entities, 500))
dataset.extend(generate_intent_data(PAYMENT_INFO_PATTERNS, "payment_info", entities, 500))
dataset.extend(generate_intent_data(PRODUCT_PRICE_PATTERNS, "product_price", entities, 500))
dataset.extend(generate_intent_data(PRODUCT_STOCK_PATTERNS, "product_stock", entities, 500))
dataset.extend(generate_intent_data(PRODUCT_DESCRIPTION_PATTERNS, "product_description", entities, 500))
dataset.extend(generate_intent_data(OUT_OF_SCOPE_PATTERNS, "out_of_scope", entities, 500))

# Shuffle
random.shuffle(dataset)

# Save
df = pd.DataFrame(dataset)
df.to_csv('data/synthetic/intent_dataset_v2.csv', index=False)

print(f"Generated {len(dataset)} samples")
print(f"\nDistribution:\n{df['intent'].value_counts()}")
print(f"\nNoise level distribution:\n{df['noise_level'].value_counts()}")

# Show samples per intent
for intent in df['intent'].unique():
    print(f"\n=== {intent.upper()} ===")
    samples = df[df['intent'] == intent]['text'].head(5).tolist()
    for s in samples:
        print(f"  - {s}")
```

### Success Criteria:

#### Automated Verification:
- [ ] Baseline evaluation saved: `./evaluation/baseline_v1_results.csv`
- [ ] New dataset generated: `./data/synthetic/intent_dataset_v2.csv`
- [ ] Dataset has 3000 samples (500 per intent)
- [ ] All 6 intents present in dataset
- [ ] Noise distribution: 30% low, 40% medium, 20% high, 10% extreme

#### Manual Verification:
- [ ] Review 20 random samples from each intent for quality
- [ ] Verify out_of_scope samples cover refund, complaints, store info
- [ ] Verify product names include Bagisto products (beanie, scarf, gloves)

**Implementation Note**: After completing this phase, pause for manual review of generated data quality before proceeding to Phase 2.

---

## Phase 2: Intent Classifier v2 Training

### Overview
Train new intent classifier with 6 intents, implement confidence thresholding, and generate comparison metrics.

### Changes Required:

#### 2.1 Training Script for v2 Model
**File**: `notebook/03_train_intent_classifier_v2.ipynb`

```python
# Cell 1: Setup and data loading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
import json
import os

print("Intent Classifier v2 Training")
print("="*60)

# Load v2 dataset
df = pd.read_csv('data/synthetic/intent_dataset_v2.csv')
df = df.dropna()
df = df[df['text'].str.len() > 0]

print(f"Total samples: {len(df)}")
print(f"\nIntent distribution:\n{df['intent'].value_counts()}")

# Create label mappings for 6 intents
INTENTS = [
    "order_status",
    "payment_info",
    "product_price",
    "product_stock",
    "product_description",
    "out_of_scope"
]

label2id = {intent: idx for idx, intent in enumerate(INTENTS)}
id2label = {idx: intent for idx, intent in enumerate(INTENTS)}

df['label'] = df['intent'].map(label2id)

print(f"\nLabel mapping: {label2id}")

# Cell 2: Train/val/test split
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Cell 3: Load model and tokenizer
MODEL_NAME = "cahya/distilbert-base-indonesian"

print(f"\nLoading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,  # 6 intents now!
    id2label=id2label,
    label2id=label2id
)

# Cell 4: Tokenize datasets
def tokenize_data(df, tokenizer, max_length=128):
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['label'].tolist()
    })

train_dataset = tokenize_data(train_df, tokenizer)
val_dataset = tokenize_data(val_df, tokenizer)
test_dataset = tokenize_data(test_df, tokenizer)

print("Datasets tokenized!")

# Cell 5: Training with early stopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir="./models/intent_classifier_v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nStarting training...")
trainer.train()

# Cell 6: Save model
OUTPUT_DIR = "./models/intent_classifier_v2/best_model"
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/label_mapping.json", 'w') as f:
    json.dump({
        'label2id': label2id,
        'id2label': id2label,
        'intents': INTENTS
    }, f, indent=2)

print(f"\nModel saved to {OUTPUT_DIR}")

# Cell 7: Evaluate on test set
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

all_preds = []
all_labels = []
all_confidences = []

for item in test_dataset:
    input_ids = torch.tensor([item['input_ids']]).to(device)
    attention_mask = torch.tensor([item['attention_mask']]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    all_preds.append(pred)
    all_labels.append(item['labels'])
    all_confidences.append(confidence)

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=INTENTS))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save test results
test_results = pd.DataFrame({
    'text': test_df['text'].tolist(),
    'true_label': [id2label[l] for l in all_labels],
    'predicted_label': [id2label[p] for p in all_preds],
    'confidence': all_confidences,
    'correct': [t == p for t, p in zip(all_labels, all_preds)]
})
test_results.to_csv('./models/intent_classifier_v2/test_results.csv', index=False)

accuracy = np.mean([t == p for t, p in zip(all_labels, all_preds)])
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

#### 2.2 Confidence Threshold Analysis
**File**: `notebook/04_confidence_threshold_analysis.ipynb`

```python
# Analyze confidence scores to find optimal threshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load test results
results_v2 = pd.read_csv('./models/intent_classifier_v2/test_results.csv')

# Analyze confidence by correctness
correct = results_v2[results_v2['correct'] == True]
wrong = results_v2[results_v2['correct'] == False]

print("Confidence Statistics:")
print(f"Correct predictions - Mean: {correct['confidence'].mean():.3f}, Min: {correct['confidence'].min():.3f}")
print(f"Wrong predictions - Mean: {wrong['confidence'].mean():.3f}, Max: {wrong['confidence'].max():.3f}")

# Find optimal threshold
thresholds = np.arange(0.5, 0.95, 0.05)

for thresh in thresholds:
    above_thresh = results_v2[results_v2['confidence'] >= thresh]
    below_thresh = results_v2[results_v2['confidence'] < thresh]

    if len(above_thresh) > 0:
        acc_above = above_thresh['correct'].mean()
        coverage = len(above_thresh) / len(results_v2)

        print(f"Threshold {thresh:.2f}: Accuracy={acc_above:.3f}, Coverage={coverage:.3f}")

# Recommendation: Use threshold where accuracy > 90% with reasonable coverage
RECOMMENDED_THRESHOLD = 0.70  # Adjust based on analysis
print(f"\nRecommended threshold: {RECOMMENDED_THRESHOLD}")
```

#### 2.3 Comparison Visualization
**File**: `notebook/05_v1_vs_v2_comparison.ipynb`

```python
# Generate comparison charts for thesis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load both results
baseline_v1 = pd.read_csv('./evaluation/baseline_v1_results.csv')
results_v2 = pd.read_csv('./models/intent_classifier_v2/test_results.csv')

# Also run v2 model on real-world test cases
from transformers import pipeline

classifier_v2 = pipeline(
    "text-classification",
    model="./models/intent_classifier_v2/best_model",
    device=-1
)

# Real-world test (same as baseline)
REAL_WORLD_TESTS = [
    # ... (same test cases as Phase 1)
]

v2_real_results = []
for test in REAL_WORLD_TESTS:
    pred = classifier_v2(test["text"])[0]
    v2_real_results.append({
        "text": test["text"],
        "expected": test["expected"],
        "predicted": pred["label"],
        "confidence": pred["score"],
        "category": test["category"],
    })

v2_real_df = pd.DataFrame(v2_real_results)

# Calculate metrics
def calc_metrics(df, expected_col='expected', predicted_col='predicted'):
    in_scope = df[df[expected_col].notna()]
    correct = (in_scope[expected_col] == in_scope[predicted_col]).sum()
    total = len(in_scope)
    return correct / total if total > 0 else 0

v1_synthetic_acc = baseline_v1['correct'].mean()
v1_real_acc = 0.7143  # From original evaluation

# Map v2 predictions - handle new intents
# For comparison, map product_price/stock/description back to product_info
def map_v2_to_v1(label):
    if label in ['product_price', 'product_stock', 'product_description']:
        return 'product_info'
    return label

v2_real_df['predicted_mapped'] = v2_real_df['predicted'].apply(map_v2_to_v1)
v2_real_acc = calc_metrics(v2_real_df, 'expected', 'predicted_mapped')

# Create comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Accuracy comparison
ax1 = axes[0]
metrics = ['Synthetic Test', 'Real-World']
v1_scores = [v1_synthetic_acc, v1_real_acc]
v2_scores = [0.99, v2_real_acc]  # v2 synthetic will be calculated

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, v1_scores, width, label='Model v1 (3 intents)', color='#3498db')
bars2 = ax1.bar(x + width/2, v2_scores, width, label='Model v2 (6 intents)', color='#2ecc71')

ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison: v1 vs v2')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.set_ylim([0, 1.1])

for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}', ha='center', va='bottom', fontsize=10)

# Chart 2: Performance by category
ax2 = axes[1]
categories = ['Easy', 'Typos', 'Verbose', 'Ambiguous']
# ... calculate per-category accuracy

plt.tight_layout()
plt.savefig('./evaluation/figures/v1_vs_v2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison chart saved!")
```

### Success Criteria:

#### Automated Verification:
- [ ] Model v2 saved: `./models/intent_classifier_v2/best_model/`
- [ ] Label mapping has 6 intents: `label_mapping.json`
- [ ] Test accuracy >= 95% on synthetic data
- [ ] Comparison charts generated: `./evaluation/figures/`

#### Manual Verification:
- [ ] Model correctly classifies out_of_scope queries
- [ ] Confidence threshold effectively separates correct/wrong predictions
- [ ] v2 real-world accuracy > v1 (target: >80%)

**Implementation Note**: After completing this phase, verify that v2 outperforms v1 on real-world queries before proceeding.

---

## Phase 3: Entity Extraction Improvement

### Overview
Replace hardcoded keyword matching with semantic similarity search against Bagisto product database.

### Changes Required:

#### 3.1 Semantic Product Matcher
**File**: `src/entity_extractor.py` (NEW FILE)

```python
"""
Semantic Entity Extractor
Uses sentence embeddings to match user queries to products in database
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple
import re

class SemanticProductMatcher:
    """
    Extract product entities using semantic similarity
    instead of hardcoded keywords
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize with multilingual sentence transformer

        Args:
            model_name: HuggingFace model for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.product_embeddings = None
        self.products = []
        self.similarity_threshold = 0.5

    def load_products_from_db(self, db_handler) -> None:
        """
        Load products from Bagisto database and create embeddings

        Args:
            db_handler: BagistoQueryHandler instance
        """
        # Get all products from database
        query = """
            SELECT DISTINCT
                pf.product_id,
                pf.sku,
                pf.name,
                pf.short_description
            FROM product_flat pf
            WHERE pf.status = 1
              AND pf.visible_individually = 1
              AND pf.name IS NOT NULL
        """

        result = db_handler._execute_query(query)

        if not result:
            print("Warning: No products found in database")
            return

        self.products = []
        product_texts = []

        for row in result:
            product = {
                'id': row['product_id'],
                'sku': row['sku'],
                'name': row['name'],
                'description': row['short_description'] or ''
            }
            self.products.append(product)

            # Create searchable text: name + keywords from description
            search_text = f"{product['name']} {product['description'][:100]}"
            product_texts.append(search_text)

        # Create embeddings for all products
        print(f"Creating embeddings for {len(self.products)} products...")
        self.product_embeddings = self.model.encode(product_texts, normalize_embeddings=True)
        print("Product embeddings ready!")

    def load_products_from_list(self, products: List[Dict]) -> None:
        """
        Load products from a list (for testing without DB)

        Args:
            products: List of product dicts with 'name' and optional 'description'
        """
        self.products = products
        product_texts = [
            f"{p['name']} {p.get('description', '')[:100]}"
            for p in products
        ]
        self.product_embeddings = self.model.encode(product_texts, normalize_embeddings=True)

    def extract_product(self, query: str, top_k: int = 1) -> List[Dict]:
        """
        Extract product entities from user query using semantic similarity

        Args:
            query: User input text
            top_k: Number of top matches to return

        Returns:
            List of matched products with similarity scores
        """
        if self.product_embeddings is None or len(self.products) == 0:
            return []

        # Clean query
        query = self._clean_query(query)

        # Get query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # Calculate cosine similarity
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
        # Remove common question words
        stopwords = [
            'berapa', 'harga', 'price', 'stok', 'stock', 'ada', 'tidak',
            'gimana', 'bagaimana', 'apakah', 'bisa', 'mau', 'tanya',
            'info', 'detail', 'spek', 'spesifikasi', 'deskripsi'
        ]

        query_lower = query.lower()
        for word in stopwords:
            query_lower = query_lower.replace(word, ' ')

        # Remove extra spaces
        query_lower = ' '.join(query_lower.split())

        return query_lower


class EntityExtractor:
    """
    Main entity extraction class combining multiple extractors
    """

    def __init__(self, db_handler=None):
        """
        Initialize entity extractor

        Args:
            db_handler: Optional BagistoQueryHandler for loading products
        """
        self.product_matcher = SemanticProductMatcher()

        if db_handler:
            self.product_matcher.load_products_from_db(db_handler)

    def extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from text using regex patterns"""
        patterns = [
            r'#(\d{1,6})',
            r'(?:order|pesanan|orderan)\s*(?:nomor|no|id)?\s*[#]?(\d{1,6})',
            r'(?:nomor|no)\s*[#]?(\d{1,6})',
            r'ORD[_-]?(\w+)',
            r'\b(\d{4,6})\b',  # Standalone 4-6 digit number
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_payment_method(self, text: str) -> Optional[str]:
        """Extract payment method from text"""
        payment_keywords = {
            'gopay': ['gopay', 'go pay', 'go-pay'],
            'ovo': ['ovo'],
            'dana': ['dana'],
            'shopeepay': ['shopeepay', 'shopee pay', 'spay'],
            'linkaja': ['linkaja', 'link aja'],
            'cod': ['cod', 'cash on delivery', 'bayar di tempat'],
            'transfer': ['transfer', 'tf', 'transfer bank'],
            'bca': ['bca'],
            'mandiri': ['mandiri'],
            'qris': ['qris'],
            'credit_card': ['kartu kredit', 'credit card', 'cc'],
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
            matches = self.product_matcher.extract_product(text, top_k=3)
            if matches:
                entities['product_matches'] = matches
                entities['product_name'] = matches[0]['product']['name']
                entities['product_id'] = matches[0]['product']['id']

        return entities
```

#### 3.2 Update Chatbot to Use New Entity Extractor
**File**: `src/chatbot.py` (MODIFY)

```python
# Add import
from src.entity_extractor import EntityExtractor

class EcommerceChatbot:

    def __init__(self, model_path, db_config=None):
        print("Loading chatbot...")

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
            self.confidence_threshold = label_mapping.get('confidence_threshold', 0.7)

        # Initialize query handler
        self.query_handler = BagistoQueryHandler(db_config)

        # Initialize NEW entity extractor with semantic matching
        self.entity_extractor = EntityExtractor(db_handler=self.query_handler)

        # Initialize response generator
        self.response_generator = TemplateResponseGenerator()

        print("Chatbot ready!")

    def process_query(self, user_query: str) -> Dict:
        """Main chatbot pipeline with confidence thresholding"""

        # Step 1: Intent Classification
        intent_result = self.intent_classifier(user_query)[0]
        intent = intent_result['label']
        confidence = intent_result['score']

        print(f"Intent: {intent} (confidence: {confidence:.3f})")

        # Step 2: Confidence threshold check
        if confidence < self.confidence_threshold:
            return {
                'intent': 'fallback',
                'confidence': confidence,
                'entities': {},
                'response': self.response_generator.generate_fallback(
                    "low_confidence",
                    user_query
                )
            }

        # Step 3: Check for out_of_scope intent
        if intent == 'out_of_scope':
            return {
                'intent': 'out_of_scope',
                'confidence': confidence,
                'entities': {},
                'response': self.response_generator.generate_fallback(
                    "out_of_scope",
                    user_query
                )
            }

        # Step 4: Entity Extraction using semantic matching
        entities = self.entity_extractor.extract_entities(user_query, intent)

        print(f"Entities: {entities}")

        # Step 5: Query Database
        db_result = self._query_database(intent, entities)

        print(f"DB Result: {db_result}")

        # Step 6: Generate Response
        response = self.response_generator.generate(intent, db_result, user_query)

        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'response': response
        }

    def _query_database(self, intent: str, entities: Dict) -> Dict:
        """Route to appropriate database query based on intent"""

        if intent == 'order_status':
            return self.query_handler.get_order_status(
                order_id=entities.get('order_id')
            )

        elif intent == 'payment_info':
            return self.query_handler.get_payment_methods()

        elif intent in ['product_price', 'product_stock', 'product_description']:
            # Use product_id from semantic matching if available
            if 'product_id' in entities:
                return self.query_handler.get_product_by_id(
                    product_id=entities['product_id'],
                    query_type=intent  # price, stock, or description
                )
            elif 'product_name' in entities:
                return self.query_handler.get_product_info(
                    product_name=entities['product_name']
                )
            else:
                return {
                    'found': False,
                    'message': 'Produk apa yang ingin Anda tanyakan?'
                }

        return {'error': 'Unknown intent'}
```

### Success Criteria:

#### Automated Verification:
- [ ] `src/entity_extractor.py` created and imports successfully
- [ ] Semantic matcher loads products from Bagisto DB
- [ ] Product matching returns results with similarity scores
- [ ] Unit tests pass for entity extraction

#### Manual Verification:
- [ ] Query "harga arctic beanie" matches "Arctic Cozy Knit Unisex Beanie"
- [ ] Query "stok sarung tangan" matches "Arctic Touchscreen Winter Gloves"
- [ ] Confidence threshold triggers fallback for ambiguous queries

**Implementation Note**: Test semantic matching with actual Bagisto products before proceeding.

---

## Phase 4: Self-Hosted LLM Text-to-SQL Integration

### Overview
Add LLM-based SQL generation using **self-hosted open-source models**. We'll use lightweight models (~7B parameters) that can run on medium-spec hardware without cloud dependencies.

### Model Options (Choose One):

| Model | Size | Best For | Hardware Requirement |
|-------|------|----------|---------------------|
| **SQLCoder-7B-2** | 7B | Text-to-SQL specifically | GPU 8GB+ VRAM or CPU 16GB+ RAM |
| **Mistral-7B-Instruct** | 7B | General + good SQL | GPU 8GB+ VRAM or CPU 16GB+ RAM |
| **Phi-3-Mini** | 3.8B | Lighter, still capable | GPU 6GB+ VRAM or CPU 8GB+ RAM |
| **Llama-3-8B-Instruct** | 8B | Best multilingual | GPU 10GB+ VRAM or CPU 16GB+ RAM |

**Recommended**: `defog/sqlcoder-7b-2` - specifically fine-tuned for Text-to-SQL tasks.

### Inference Options:

1. **Ollama** (Easiest) - Simple CLI tool for running LLMs locally
2. **llama.cpp** (Most flexible) - C++ inference with GGUF quantization
3. **vLLM** (Best for GPU) - High-throughput inference server

### Changes Required:

#### 4.0 Install Local LLM Runtime
**Option A: Using Ollama (Recommended for simplicity)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull SQLCoder model (or Mistral)
ollama pull sqlcoder:7b
# OR
ollama pull mistral:7b-instruct

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

**Option B: Using llama.cpp with GGUF models**

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Download quantized model (Q4_K_M for balance of speed/quality)
# From: https://huggingface.co/TheBloke/sqlcoder-7B-2-GGUF
wget https://huggingface.co/TheBloke/sqlcoder-7B-2-GGUF/resolve/main/sqlcoder-7b-2.Q4_K_M.gguf

# Start server
./server -m sqlcoder-7b-2.Q4_K_M.gguf -c 2048 --host 0.0.0.0 --port 8080
```

#### 4.1 LLM Query Generator (Self-Hosted)
**File**: `src/llm_query_generator.py` (NEW FILE)

```python
"""
Self-Hosted LLM Text-to-SQL Generator for Bagisto E-commerce
Uses local open-source models (Ollama/llama.cpp) - NO cloud dependencies

Supported backends:
- Ollama (default, easiest)
- llama.cpp server
- vLLM (for GPU inference)
"""

import os
import json
import re
import requests
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class OllamaBackend(LLMBackend):
    """
    Ollama backend for local LLM inference
    https://ollama.com/
    """

    def __init__(
        self,
        model: str = "sqlcoder:7b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Generate completion using Ollama"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.1,  # Low temperature for deterministic SQL
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Ollama error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ollama request failed: {e}")
            return None


class LlamaCppBackend(LLMBackend):
    """
    llama.cpp server backend
    https://github.com/ggerganov/llama.cpp
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.api_url = f"{base_url}/completion"

    def is_available(self) -> bool:
        """Check if llama.cpp server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Generate completion using llama.cpp server"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": 0.1,
                    "stop": [";", "\n\n", "```"],
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("content", "")
            else:
                print(f"llama.cpp error: {response.status_code}")
                return None

        except Exception as e:
            print(f"llama.cpp request failed: {e}")
            return None


class LocalLLMQueryGenerator:
    """
    Generate SQL queries from natural language using self-hosted LLM
    with Bagisto schema context
    """

    # Bagisto schema context (optimized for smaller models)
    SCHEMA_PROMPT = """### Database Schema:

CREATE TABLE product_flat (
    product_id INT PRIMARY KEY,
    sku VARCHAR(255),
    name VARCHAR(255),
    price DECIMAL(12,4),
    short_description TEXT,
    status TINYINT DEFAULT 1,
    visible_individually TINYINT DEFAULT 1
);

CREATE TABLE product_inventories (
    product_id INT,
    qty INT DEFAULT 0
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    increment_id VARCHAR(255),
    status VARCHAR(50),
    customer_first_name VARCHAR(255),
    customer_last_name VARCHAR(255),
    grand_total DECIMAL(12,4),
    created_at TIMESTAMP
);

### Rules:
- Only SELECT queries
- Always add LIMIT 10
- For products: WHERE status = 1 AND visible_individually = 1
- Use LIKE '%keyword%' for text search

### Task: Generate SQL for this question:
{question}

### SQL:
SELECT"""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "sqlcoder:7b",
        base_url: Optional[str] = None
    ):
        """
        Initialize local LLM query generator

        Args:
            backend: "ollama" or "llamacpp"
            model: Model name (for Ollama)
            base_url: Custom server URL
        """
        if backend == "ollama":
            url = base_url or "http://localhost:11434"
            self.llm = OllamaBackend(model=model, base_url=url)
        elif backend == "llamacpp":
            url = base_url or "http://localhost:8080"
            self.llm = LlamaCppBackend(base_url=url)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend_name = backend
        self._check_availability()

    def _check_availability(self):
        """Check if LLM backend is available"""
        if self.llm.is_available():
            print(f"Local LLM ({self.backend_name}) is ready!")
        else:
            print(f"Warning: Local LLM ({self.backend_name}) not available.")
            print("Make sure the server is running:")
            print("  Ollama: ollama serve")
            print("  llama.cpp: ./server -m model.gguf")

    def generate_sql(self, user_query: str, intent: str) -> Tuple[Optional[str], bool]:
        """
        Generate SQL query from natural language

        Args:
            user_query: User's question in Indonesian
            intent: Classified intent

        Returns:
            Tuple of (sql_query, is_safe)
        """
        if not self.llm.is_available():
            return None, False

        # Build prompt with schema context
        prompt = self.SCHEMA_PROMPT.format(question=user_query)

        # Generate SQL
        response = self.llm.generate(prompt, max_tokens=200)

        if not response:
            return None, False

        # Complete the SELECT statement
        sql = "SELECT " + response.strip()

        # Sanitize and validate
        sql = self._sanitize_sql(sql)
        is_safe = self._validate_sql(sql)

        if is_safe:
            print(f"Generated SQL: {sql}")

        return sql, is_safe

    def _sanitize_sql(self, sql: str) -> str:
        """Clean up generated SQL"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)

        # Remove everything after first semicolon (prevent injection)
        if ';' in sql:
            sql = sql.split(';')[0]

        # Remove trailing whitespace and newlines
        sql = sql.strip()

        # Ensure LIMIT exists
        if 'LIMIT' not in sql.upper():
            sql = sql + ' LIMIT 10'

        return sql

    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL query is safe to execute"""
        if not sql or len(sql) < 10:
            return False

        sql_upper = sql.upper()

        # Block dangerous operations
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE',
            'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC',
            'EXECUTE', '--', '/*', 'UNION'
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                print(f"Blocked dangerous SQL: {keyword}")
                return False

        # Must start with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False

        return True

    def execute_safe_query(self, sql: str, db_handler) -> Optional[Dict]:
        """Execute validated SQL query"""
        if not self._validate_sql(sql):
            return None

        try:
            result = db_handler._execute_query(sql)
            return {
                'success': True,
                'data': result,
                'count': len(result) if result else 0
            }
        except Exception as e:
            print(f"SQL Execution Error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class HybridQueryEngine:
    """
    Combines rule-based queries with local LLM fallback
    """

    def __init__(
        self,
        db_handler,
        llm_backend: str = "ollama",
        llm_model: str = "sqlcoder:7b"
    ):
        self.db_handler = db_handler
        self.llm = LocalLLMQueryGenerator(
            backend=llm_backend,
            model=llm_model
        )

    def query(self, intent: str, entities: Dict, user_query: str) -> Dict:
        """
        Execute query using best available method

        Priority:
        1. Direct DB query if entities extracted
        2. LLM-generated SQL if entities unclear
        3. Fallback to featured products
        """

        # Try direct query first
        if intent == 'product_price' and 'product_id' in entities:
            return self.db_handler.get_product_price(entities['product_id'])

        if intent == 'product_stock' and 'product_id' in entities:
            return self.db_handler.get_product_stock(product_id=entities['product_id'])

        if intent == 'product_description' and 'product_id' in entities:
            return self.db_handler.get_product_description(entities['product_id'])

        # Try LLM if entities not clear
        if self.llm.llm.is_available() and intent.startswith('product_'):
            sql, is_safe = self.llm.generate_sql(user_query, intent)

            if is_safe and sql:
                result = self.llm.execute_safe_query(sql, self.db_handler)
                if result and result.get('success'):
                    return {
                        'found': True,
                        'source': 'llm',
                        'data': result['data']
                    }

        # Fallback to featured products
        return self.db_handler.get_product_info(product_name=None)
```

#### 4.2 Setup Script for Local LLM
**File**: `scripts/setup_local_llm.sh` (NEW FILE)

```bash
#!/bin/bash
# Setup script for local LLM inference

echo "=== Local LLM Setup for E-commerce Chatbot ==="

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "Ollama is installed"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull recommended model
echo ""
echo "Pulling SQLCoder 7B model (best for Text-to-SQL)..."
echo "This may take a while (~4GB download)..."
ollama pull sqlcoder:7b

# Alternative: Mistral for general use
# echo "Pulling Mistral 7B model..."
# ollama pull mistral:7b-instruct

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the LLM server, run:"
echo "  ollama serve"
echo ""
echo "To test the model:"
echo "  ollama run sqlcoder:7b 'SELECT * FROM products WHERE price < 100'"
```

#### 4.3 Test Local LLM
**File**: `notebook/04_test_local_llm.ipynb`

```python
# Cell 1: Test Ollama connection
import requests

def test_ollama():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Ollama is running!")
            print(f"Available models: {[m['name'] for m in models]}")
            return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        print("\nTo start Ollama:")
        print("  1. Open terminal")
        print("  2. Run: ollama serve")
        print("  3. In another terminal: ollama pull sqlcoder:7b")
        return False

test_ollama()

# Cell 2: Test SQL generation
from src.llm_query_generator import LocalLLMQueryGenerator

# Initialize with Ollama backend
llm = LocalLLMQueryGenerator(backend="ollama", model="sqlcoder:7b")

# Test queries in Indonesian
test_queries = [
    ("Tampilkan semua produk", "product_description"),
    ("Produk dengan harga di bawah 50000", "product_price"),
    ("Stok produk beanie", "product_stock"),
    ("Produk paling murah", "product_price"),
]

print("Testing SQL Generation:")
print("=" * 60)

for query, intent in test_queries:
    print(f"\nQuery: {query}")
    print(f"Intent: {intent}")

    sql, is_safe = llm.generate_sql(query, intent)

    if is_safe:
        print(f"SQL: {sql}")
    else:
        print("Failed to generate safe SQL")
```

#### 4.2 Update Query Handler with New Methods
**File**: `src/query_handler.py` (ADD METHODS)

```python
# Add these methods to BagistoQueryHandler class

def get_product_by_id(self, product_id: int, query_type: str = 'all') -> Dict:
    """
    Get specific product information by ID

    Args:
        product_id: Product ID
        query_type: 'price', 'stock', 'description', or 'all'
    """
    try:
        if query_type == 'price':
            query = """
                SELECT name, price, special_price
                FROM product_flat
                WHERE product_id = %s AND status = 1
                LIMIT 1
            """
        elif query_type == 'stock':
            query = """
                SELECT pf.name, COALESCE(SUM(pi.qty), 0) as stock
                FROM product_flat pf
                LEFT JOIN product_inventories pi ON pf.product_id = pi.product_id
                WHERE pf.product_id = %s AND pf.status = 1
                GROUP BY pf.name
            """
        elif query_type == 'description':
            query = """
                SELECT name, short_description, description
                FROM product_flat
                WHERE product_id = %s AND status = 1
                LIMIT 1
            """
        else:
            query = """
                SELECT
                    pf.product_id, pf.name, pf.price, pf.short_description,
                    COALESCE(SUM(pi.qty), 0) as stock
                FROM product_flat pf
                LEFT JOIN product_inventories pi ON pf.product_id = pi.product_id
                WHERE pf.product_id = %s AND pf.status = 1
                GROUP BY pf.product_id, pf.name, pf.price, pf.short_description
            """

        result = self._execute_query(query, (product_id,))

        if result and len(result) > 0:
            return {
                'found': True,
                'product': result[0],
                'query_type': query_type
            }

        return {'found': False, 'message': 'Product not found'}

    except Exception as e:
        return {'error': str(e)}

def get_product_price(self, product_id: int) -> Dict:
    """Get product price only"""
    return self.get_product_by_id(product_id, 'price')

def get_product_description(self, product_id: int) -> Dict:
    """Get product description only"""
    return self.get_product_by_id(product_id, 'description')
```

### Success Criteria:

#### Automated Verification:
- [ ] `src/llm_query_generator.py` created
- [ ] SQL validation blocks dangerous queries
- [ ] LLM generates valid SELECT queries
- [ ] Hybrid engine falls back correctly when LLM unavailable

#### Manual Verification:
- [ ] Query "produk paling murah" generates correct SQL
- [ ] Query "semua produk di bawah 50000" works
- [ ] LLM errors don't crash the system (graceful fallback)

**Implementation Note**: LLM integration requires OpenAI API key. Set `OPENAI_API_KEY` environment variable or pass directly.

---

## Phase 5: Evaluation Suite & Thesis Documentation

### Overview
Create comprehensive evaluation with ablation studies, generate all comparison charts, and document for thesis.

### Changes Required:

#### 5.1 Comprehensive Evaluation Script
**File**: `notebook/06_comprehensive_evaluation.ipynb`

```python
# Cell 1: Setup comprehensive test suite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Create comprehensive test cases
COMPREHENSIVE_TEST_SUITE = {
    "easy": [
        # Clear, well-formed queries
        {"text": "cek pesanan 12345", "expected": "order_status"},
        {"text": "status order saya", "expected": "order_status"},
        {"text": "bisa bayar pakai gopay", "expected": "payment_info"},
        {"text": "metode pembayaran apa saja", "expected": "payment_info"},
        {"text": "harga sepatu berapa", "expected": "product_price"},
        {"text": "stok laptop ada", "expected": "product_stock"},
        {"text": "deskripsi produk beanie", "expected": "product_description"},
        {"text": "mau refund", "expected": "out_of_scope"},
    ],
    "moderate_typo": [
        # Common typos
        {"text": "cek psanan 123", "expected": "order_status"},
        {"text": "byr pake apa", "expected": "payment_info"},
        {"text": "hrga laptop brp", "expected": "product_price"},
        {"text": "stok septu ada g", "expected": "product_stock"},
    ],
    "extreme_typo": [
        # Heavy abbreviations and typos
        {"text": "ordr sy mn", "expected": "order_status"},
        {"text": "byr pk ap", "expected": "payment_info"},
        {"text": "hrg brp", "expected": "product_price"},
        {"text": "ad stok g", "expected": "product_stock"},
    ],
    "slang": [
        # Indonesian internet slang
        {"text": "brg gw mna bro", "expected": "order_status"},
        {"text": "bs gpay ga sih", "expected": "payment_info"},
        {"text": "brp duit laptop", "expected": "product_price"},
        {"text": "ready ga stoknya", "expected": "product_stock"},
    ],
    "verbose": [
        # Long, conversational queries
        {"text": "halo min mau tanya dong pesanan saya yang kemarin itu udah dikirim belum ya makasih", "expected": "order_status"},
        {"text": "maaf mau nanya nih untuk pembayarannya bisa pakai metode apa aja ya kak", "expected": "payment_info"},
        {"text": "permisi kak saya tertarik sama produk beanie itu harganya berapa ya", "expected": "product_price"},
    ],
    "ambiguous": [
        # Multi-intent or unclear queries
        {"text": "mau beli sepatu bisa cod ga", "expected": "payment_info"},
        {"text": "orderan belum bayar", "expected": "order_status"},
        {"text": "harga berapa kalau bayar cash", "expected": "product_price"},
    ],
    "out_of_scope": [
        # Should trigger fallback
        {"text": "mau komplain barang rusak", "expected": "out_of_scope"},
        {"text": "gimana cara refund", "expected": "out_of_scope"},
        {"text": "jam buka toko kapan", "expected": "out_of_scope"},
        {"text": "alamat toko dimana", "expected": "out_of_scope"},
        {"text": "mau cancel pesanan", "expected": "out_of_scope"},
        {"text": "lowongan kerja ada ga", "expected": "out_of_scope"},
    ],
}

# Flatten test suite
all_tests = []
for category, tests in COMPREHENSIVE_TEST_SUITE.items():
    for test in tests:
        test['category'] = category
        all_tests.append(test)

print(f"Total test cases: {len(all_tests)}")
for cat, tests in COMPREHENSIVE_TEST_SUITE.items():
    print(f"  {cat}: {len(tests)}")

# Cell 2: Run evaluation on v2 model
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/intent_classifier_v2/best_model",
    device=-1
)

# Map v2 intents
INTENT_MAPPING = {
    'order_status': 'order_status',
    'payment_info': 'payment_info',
    'product_price': 'product_price',
    'product_stock': 'product_stock',
    'product_description': 'product_description',
    'out_of_scope': 'out_of_scope',
}

results = []
for test in all_tests:
    pred = classifier(test['text'])[0]

    results.append({
        'text': test['text'],
        'expected': test['expected'],
        'predicted': pred['label'],
        'confidence': pred['score'],
        'category': test['category'],
        'correct': pred['label'] == test['expected']
    })

results_df = pd.DataFrame(results)

# Cell 3: Generate metrics
print("="*60)
print("COMPREHENSIVE EVALUATION RESULTS")
print("="*60)

# Overall accuracy
overall_acc = results_df['correct'].mean()
print(f"\nOverall Accuracy: {overall_acc:.2%}")

# Per-category accuracy
print("\nPer-Category Accuracy:")
for category in COMPREHENSIVE_TEST_SUITE.keys():
    cat_df = results_df[results_df['category'] == category]
    cat_acc = cat_df['correct'].mean()
    print(f"  {category}: {cat_acc:.2%} ({cat_df['correct'].sum()}/{len(cat_df)})")

# Per-intent metrics
print("\nClassification Report:")
print(classification_report(
    results_df['expected'],
    results_df['predicted'],
    zero_division=0
))

# Cell 4: Generate thesis figures
os.makedirs('./thesis_figures', exist_ok=True)

# Figure 1: Category Performance Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
categories = list(COMPREHENSIVE_TEST_SUITE.keys())
accuracies = [
    results_df[results_df['category'] == cat]['correct'].mean()
    for cat in categories
]

colors = plt.cm.RdYlGn(np.array(accuracies))
bars = ax.bar(categories, accuracies, color=colors, edgecolor='black')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Query Category', fontsize=12)
ax.set_title('Intent Classification Accuracy by Query Category', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.axhline(y=0.8, color='red', linestyle='--', label='Target (80%)')

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('./thesis_figures/category_accuracy.png', dpi=300)
plt.close()

# Figure 2: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(results_df['expected'], results_df['predicted'])
labels = sorted(results_df['expected'].unique())

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Confusion Matrix - Intent Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./thesis_figures/confusion_matrix.png', dpi=300)
plt.close()

# Figure 3: Confidence Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

correct_conf = results_df[results_df['correct']]['confidence']
wrong_conf = results_df[~results_df['correct']]['confidence']

axes[0].hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
axes[0].hist(wrong_conf, bins=20, alpha=0.7, label='Wrong', color='red')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Confidence Distribution')
axes[0].legend()

axes[1].boxplot([correct_conf, wrong_conf], labels=['Correct', 'Wrong'])
axes[1].set_ylabel('Confidence Score')
axes[1].set_title('Confidence by Correctness')

plt.tight_layout()
plt.savefig('./thesis_figures/confidence_analysis.png', dpi=300)
plt.close()

print("\nFigures saved to ./thesis_figures/")
```

#### 5.2 Ablation Study Script
**File**: `notebook/07_ablation_study.ipynb`

```python
# Ablation study: measure impact of each component

# Components to ablate:
# 1. Improved synthetic data (v2 vs v1 data)
# 2. Additional intents (6 vs 3)
# 3. Confidence thresholding
# 4. Semantic entity extraction
# 5. LLM Text-to-SQL

ablation_results = {
    "Full System (v2)": {"accuracy": 0.0, "description": "All improvements"},
    "Without LLM": {"accuracy": 0.0, "description": "Semantic matching only, no LLM"},
    "Without Semantic Matching": {"accuracy": 0.0, "description": "Keyword matching only"},
    "Without Confidence Threshold": {"accuracy": 0.0, "description": "No fallback on low confidence"},
    "3 Intents Only": {"accuracy": 0.0, "description": "Original intent structure"},
    "Original v1": {"accuracy": 0.0, "description": "Baseline system"},
}

# Run each configuration and measure accuracy
# ... (implementation for each ablation)

# Generate ablation chart
fig, ax = plt.subplots(figsize=(12, 6))
configs = list(ablation_results.keys())
accuracies = [r['accuracy'] for r in ablation_results.values()]

bars = ax.barh(configs, accuracies, color='steelblue', edgecolor='black')
ax.set_xlabel('Real-World Accuracy')
ax.set_title('Ablation Study: Impact of Each Component', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1])

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{acc:.1%}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('./thesis_figures/ablation_study.png', dpi=300)
```

#### 5.3 Final Metrics Summary
**File**: `evaluation/FINAL_METRICS.md`

```markdown
# E-Commerce Chatbot - Final Evaluation Metrics

## Performance Summary

| Metric | v1 (Baseline) | v2 (Improved) | Improvement |
|--------|---------------|---------------|-------------|
| Synthetic Accuracy | 99.11% | TBD% | TBD |
| Real-World Accuracy | 71.43% | TBD% | TBD |
| Out-of-Scope Detection | 0% | TBD% | TBD |
| Fallback Trigger Rate | 0% | TBD% | TBD |

## Per-Category Performance

| Category | v1 Accuracy | v2 Accuracy |
|----------|-------------|-------------|
| Easy | TBD% | TBD% |
| Moderate Typo | TBD% | TBD% |
| Extreme Typo | TBD% | TBD% |
| Slang | TBD% | TBD% |
| Verbose | TBD% | TBD% |
| Ambiguous | TBD% | TBD% |
| Out-of-Scope | N/A | TBD% |

## Ablation Study Results

| Configuration | Accuracy | Delta from Full |
|---------------|----------|-----------------|
| Full System | TBD% | - |
| Without LLM | TBD% | TBD |
| Without Semantic | TBD% | TBD |
| Without Confidence | TBD% | TBD |
| 3 Intents Only | TBD% | TBD |
| Original v1 | 71.43% | TBD |
```

### Success Criteria:

#### Automated Verification:
- [ ] All thesis figures generated in `./thesis_figures/`
- [ ] Ablation study completed with all configurations
- [ ] Final metrics document created

#### Manual Verification:
- [ ] Real-world accuracy >85% (target)
- [ ] Figures are publication-quality
- [ ] Ablation study shows clear component contributions

---

## Testing Strategy

### Unit Tests

1. **Entity Extractor Tests**
   - Order ID extraction from various formats
   - Product name semantic matching
   - Payment method detection

2. **Intent Classifier Tests**
   - Correct classification for each intent
   - Confidence scores in expected ranges
   - Edge case handling

3. **LLM Query Generator Tests**
   - SQL validation blocks dangerous queries
   - Valid queries execute successfully
   - Fallback works when LLM unavailable

### Integration Tests

1. **End-to-End Pipeline**
   - User query → Intent → Entities → DB → Response
   - Fallback triggers correctly
   - Response format is correct

2. **Database Integration**
   - Connection handling
   - Query execution
   - Error recovery

### Manual Testing Steps

1. Test with 20 real user queries (varied difficulty)
2. Verify out-of-scope detection triggers WhatsApp redirect
3. Verify semantic matching works for ALL Bagisto products
4. Test LLM fallback with complex queries
5. Verify confidence threshold behavior

---

## Performance Considerations

1. **Semantic Matching Speed**
   - Pre-compute product embeddings on startup
   - Cache frequently queried products
   - Expected latency: <100ms for similarity search

2. **Local LLM Performance**
   - **SQLCoder 7B with Q4 quantization**: ~2-5 seconds per query on CPU, <1 second on GPU
   - **Ollama** handles model loading/caching automatically
   - Implement timeout (30 seconds for CPU, 10 seconds for GPU)
   - Fallback to rule-based if LLM unavailable or slow
   - **Hardware recommendations**:
     - Minimum: 16GB RAM (CPU inference with quantized model)
     - Recommended: GPU with 8GB+ VRAM (NVIDIA RTX 3060 or better)
     - For faster inference: Use `Q4_K_M` quantization (4-bit, ~4GB model size)

3. **Database Queries**
   - Use connection pooling
   - Index frequently queried columns
   - Limit result sets

4. **Model Loading Strategy**
   - Load intent classifier on startup (stays in memory)
   - Load embedding model on startup (for semantic matching)
   - LLM loaded by Ollama on first request (then cached)

---

## References

- Original ticket: `thoughts/tickets/firas/initial.md`
- Current notebook: `notebook/skripsi_chatbot.ipynb`
- Bagisto schema: `bagisto_database_report.json`
- Model v1: `models/intent_classifier/best_model/`
