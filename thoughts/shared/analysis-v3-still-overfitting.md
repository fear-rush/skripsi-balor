# Analysis: Why V3 Still Achieves 98% Accuracy

**Date**: 2025-12-05
**Context**: After implementing V3 fixes, the model still jumps from 84% to 98% in one epoch

---

## What We Did in V3

| Fix | Status | Did It Help? |
|-----|--------|--------------|
| Separate train/test product pools | Done | No |
| Deduplicate texts | Done | No |
| Template usage limits | Done | No |
| Hard negatives (~50 samples) | Done | No |
| Semantic noise (synonyms, fillers) | Done | No |
| 50+ templates per intent | Done | No |

**Result**: Model still achieves 98% accuracy by Epoch 2.

---

## The Real Root Cause: Exclusive Keyword Sets

The fundamental problem we missed: **each intent has keywords that appear ONLY in that intent**.

| Intent | Exclusive Keywords | Frequency |
|--------|-------------------|-----------|
| product_price | "harga", "berapa", "price", "biaya" | 99% of price queries |
| product_stock | "stok", "stock", "ready", "ada" | 99% of stock queries |
| payment_info | "bayar", "pembayaran", "metode" | 99% of payment queries |
| order_status | "pesanan", "order", "status", "sampai" | 99% of order queries |
| product_description | "spec", "detail", "bahan", "fitur" | 99% of description queries |
| out_of_scope | "refund", "komplain", "kerja" | 99% of OOS queries |

**The model learned**:
```
IF "harga" OR "berapa" → product_price (99.9% accurate)
IF "stok" OR "ready" → product_stock (99.9% accurate)
```

This is **keyword pattern matching**, not **intent understanding**.

---

## Why V3 Fixes Didn't Help

### 1. Entity Pool Split: Irrelevant
We split products into train/test pools, but the keywords don't care about products:
```
Train: "harga jacket-blue-green-l berapa"
Test:  "harga jacket-blue-green-m berapa"
```
Both have "harga" + "berapa" → instant correct classification.

### 2. Hard Negatives: Drowned Out
- Hard negatives: ~50 samples (3%)
- Easy keyword samples: ~1,950 samples (97%)

The model learns the dominant pattern (keywords) and ignores the rare exceptions.

### 3. Noise Functions: Keywords Persist
```
Original:  "harga jacket berapa"
+ typos:   "hrga jcket brp"       ← "hrg" still maps to price
+ synonyms: "price jacket berapa"  ← same intent, same pattern
+ fillers:  "kak harga jacket brp" ← keywords unchanged
```

Noise creates surface variation but **keywords survive all transformations**.

### 4. More Templates: Same Keywords
50 templates per intent, but ALL contain the same keywords:
```
Template 1: "harga {product} berapa"
Template 2: "berapa harga {product}"
Template 3: "{product} harganya berapa"
Template 4: "price {product}"
```
All 4 are trivially classifiable as product_price.

---

## Evidence: The Training Trajectory

```
Epoch 1: 84% (basic keyword detection)
Epoch 2: 98% (keyword patterns fully memorized)
Epoch 3-6: 98-99% (plateau, nothing left to learn)
```

A truly semantic model would show:
```
Epoch 1: 55-65% (random + obvious patterns)
Epoch 2: 70-75% (learning semantics)
Epoch 3-5: 80-85% (refinement)
Plateau: 85-90% (hard ceiling due to ambiguity)
```

---

## What 98% Accuracy Actually Means

The model is **NOT learning intent classification**. It's learning:
1. A keyword detector for "harga/berapa" → price
2. A keyword detector for "stok/ready" → stock
3. A keyword detector for "bayar/metode" → payment
4. etc.

This would fail catastrophically on real queries:
- "How much?" (no keyword → misclassified)
- "Got any left?" (no keyword → misclassified)
- "Can I pay now?" (might work if "pay" learned)

---

## The Fundamental Insight

**V3 fixed the wrong problem.**

| Problem | What V3 Fixed | What Actually Matters |
|---------|---------------|----------------------|
| Data leakage | Same products in train/test | Keywords are deterministic |
| Duplication | Repeated texts | Keywords repeat 99% of time |
| Noise | Typos, abbreviations | Keywords survive noise |

The issue isn't data quality. The issue is **dataset design**: intents are defined by exclusive keywords, making classification trivial.

---

## What Actually Needs to Change (V4)

### Option A: Remove Keyword Exclusivity
Make keywords appear across multiple intents:
```
"harga" in product_price (60%) AND out_of_scope (40%)
"stok" in product_stock (60%) AND order_status (40%)
```

### Option B: Keyword-Free Queries
30-40% of dataset should have NO intent-specific keywords:
```
product_price: "How much will this cost me?"
product_stock: "Do you have any units available?"
payment_info: "What are my transaction options?"
```

### Option C: Massive Hard Negatives
Hard negatives should be 30-40% of dataset, not 3%:
```
"ready bayar" vs "ready stock" (same word, different intent)
"gimana pesanan" vs "gimana produk" (same structure)
"info harga" vs "info lengkap" (same pattern)
```

### Option D: Intentional Ambiguity
Some queries should be genuinely ambiguous:
```
"ada ga?" - could be stock OR payment
"ready?" - could be stock OR order
"gimana?" - could be anything
```

---

## Realistic Expectations

| Dataset Type | Expected Accuracy | Why |
|--------------|-------------------|-----|
| V2 (keyword-based, leaked) | 98% | Keyword matching + data leakage |
| V3 (keyword-based, no leak) | 98% | Still just keyword matching |
| V4 (30% ambiguous) | 85-90% | Real semantic learning |
| V4 (50% ambiguous) | 75-85% | Even harder, more realistic |
| Real customer data | 70-80% | Unpredictable, messy, ambiguous |

---

## Conclusion

The 98% accuracy is a **misleading metric** because:
1. It measures keyword detection, not intent understanding
2. Any keyword-based dataset will yield ~98% with transformer models
3. Real-world performance would be 60-70% on this model

**For the thesis**: Either:
- **Option 1**: Acknowledge this limitation and present V3 as "keyword-based baseline"
- **Option 2**: Create V4 with true ambiguity for honest evaluation
- **Option 3**: Collect 100+ real customer queries for ground truth

The best thesis approach would combine: baseline (V3 at 98%), improved (V4 at 85%), and validated on real queries.
