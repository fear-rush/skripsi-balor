# V4 Intent Classifier Implementation Plan

**Date**: 2025-12-05
**Goal**: Create a dataset that achieves **85-90% accuracy** through true semantic learning, not keyword matching.

---

## Problem Statement

### V2 → V3 → V4 Evolution

| Version | Issue | Result |
|---------|-------|--------|
| V2 | Data leakage (same products in train/test) | 98.4% |
| V3 | Fixed leakage, but keywords exclusive per intent | 98% |
| V4 | Break keyword exclusivity + add ambiguity | 85-90% (target) |

### Root Cause of 98% in V3

Each intent has **exclusive keywords** that appear ONLY in that intent:

```
product_price: "harga", "berapa", "price"
product_stock: "stok", "stock", "ready"
payment_info: "bayar", "pembayaran", "metode"
order_status: "pesanan", "order", "status"
product_description: "spec", "detail", "bahan"
out_of_scope: "refund", "komplain"
```

Model learned: `IF "harga" THEN product_price` (99.9% accurate)

This is keyword detection, NOT intent understanding.

---

## V4 Solution Architecture

### 1. Shared Keywords (Break Exclusivity)

Keywords must appear in **multiple intents** with different contexts:

| Keyword | Intent 1 (60%) | Intent 2 (25%) | Intent 3 (15%) |
|---------|---------------|----------------|----------------|
| `berapa` | product_price ("harga berapa") | product_stock ("stok berapa") | order_status ("berapa lama") |
| `ready` | product_stock ("ready stock") | payment_info ("ready bayar") | order_status ("ready kirim") |
| `ada` | product_stock ("ada stok") | payment_info ("ada cicilan") | description ("ada fitur") |
| `gimana` | description ("produk gimana") | order_status ("pesanan gimana") | payment ("bayar gimana") |
| `info` | description ("info produk") | product_price ("info harga") | order_status ("info order") |
| `bisa` | payment_info ("bisa bayar") | product_stock ("bisa beli") | description ("bisa dipakai") |

### 2. Data Distribution

| Category | Percentage | Count | Purpose |
|----------|-----------|-------|---------|
| Keyword-based (shared) | 42% | 1,008 | Same keywords, different intents |
| Generic (no keywords) | 28% | 672 | No intent-specific keywords |
| Hard negatives | 20% | 480 | Confusing pairs |
| Ambiguous | 10% | 240 | Genuinely unclear → out_of_scope |
| **Total** | 100% | 2,400 | |

### 3. Per-Intent Sample Count

| Intent | Keyword | Generic | Hard Neg | Total |
|--------|---------|---------|----------|-------|
| order_status | 168 | 112 | 80 | 360 |
| payment_info | 168 | 112 | 80 | 360 |
| product_price | 168 | 112 | 80 | 360 |
| product_stock | 168 | 112 | 80 | 360 |
| product_description | 168 | 112 | 80 | 360 |
| out_of_scope | 168 | 112 | 80+240 | 600 |

---

## Detailed Template Design

### A. ORDER_STATUS Templates

#### Keyword-Based (shared keywords)
```python
ORDER_STATUS_KEYWORD = [
    # Classic with "pesanan/order"
    "cek pesanan {order_id}",
    "status order {order_id}",
    "tracking pesanan {order_id}",

    # Using "berapa" (shared with price)
    "pesanan berapa lama sampai",
    "order {order_id} berapa hari lagi",
    "udah berapa lama diproses",

    # Using "ready" (shared with stock/payment)
    "order ready dikirim belum",
    "pesanan ready kirim kapan",

    # Using "ada" (shared with stock)
    "ada update pesanan {order_id}",
    "ada kabar order saya",

    # Using "gimana" (shared with description)
    "pesanan saya gimana",
    "order {order_id} gimana statusnya",
]
```

#### Generic (no exclusive keywords)
```python
ORDER_STATUS_GENERIC = [
    # No "pesanan", "order", "status"
    "yang kemarin udah dikirim belum",
    "yang saya pesan udah sampai mana",
    "kapan nyampe ya",
    "kok belum dateng",
    "lama banget nih",
    "udah jalan belum",
    "sampai mana ya",
    "masih diproses",
]
```

### B. PRODUCT_PRICE Templates

#### Keyword-Based (shared keywords)
```python
PRODUCT_PRICE_KEYWORD = [
    # Classic with "harga"
    "harga {product} berapa",
    "berapa harga {product}",

    # Using "berapa" without "harga"
    "{product} berapa",
    "kena berapa {product}",
    "totalnya berapa",

    # Using "info" (shared with description)
    "info harga {product}",
    "mau tau harga {product}",

    # Using "ada" (shared with stock)
    "{product} ada diskon ga",
    "ada promo {product}",
]
```

#### Generic (no exclusive keywords)
```python
PRODUCT_PRICE_GENERIC = [
    # No "harga", "berapa", "price"
    "{product} mahal ga",
    "{product} murah ga",
    "worth it ga {product}",
    "terjangkau ga",
    "affordable ga",
    "budget friendly ga",
]
```

### C. PRODUCT_STOCK Templates

#### Keyword-Based (shared keywords)
```python
PRODUCT_STOCK_KEYWORD = [
    # Classic with "stok"
    "stok {product} ada",
    "{product} ready stock",

    # Using "ready" (shared with payment)
    "{product} ready",
    "ready ga {product}",

    # Using "ada" (shared with payment/description)
    "ada {product}",
    "{product} masih ada",

    # Using "berapa" (shared with price)
    "stok berapa",
    "{product} tinggal berapa",
    "sisa berapa unit",
]
```

#### Generic (no exclusive keywords)
```python
PRODUCT_STOCK_GENERIC = [
    # No "stok", "stock", "ready", "ada"
    "{product} kosong ga",
    "{product} habis belum",
    "bisa dapet {product}",
    "masih bisa dibeli",
    "{product} sold out",
]
```

### D. PAYMENT_INFO Templates

#### Keyword-Based (shared keywords)
```python
PAYMENT_INFO_KEYWORD = [
    # Classic with "bayar"
    "cara bayar gimana",
    "metode pembayaran apa",

    # Using "ready" (shared with stock)
    "ready bayar",
    "pembayaran ready",

    # Using "bisa" (shared with stock)
    "bisa {payment_method}",
    "bisa transfer",
    "bisa COD",

    # Using "ada" (shared with stock)
    "ada cicilan ga",
    "ada opsi {payment_method}",

    # Using "berapa" (shared with price)
    "transfer berapa",
    "biaya admin berapa",
]
```

#### Generic (no exclusive keywords)
```python
PAYMENT_INFO_GENERIC = [
    # No "bayar", "pembayaran", "metode"
    "pake apa",
    "lewat mana",
    "via apa",
    "opsi apa aja",
    "pilihan apa saja",
]
```

### E. PRODUCT_DESCRIPTION Templates

#### Keyword-Based (shared keywords)
```python
PRODUCT_DESCRIPTION_KEYWORD = [
    # Classic with "spec/detail"
    "spesifikasi {product}",
    "detail {product}",

    # Using "gimana" (shared with order)
    "{product} gimana",
    "kualitasnya gimana",

    # Using "info" (shared with price)
    "info {product}",
    "info lengkap {product}",

    # Using "ada" (shared with stock)
    "{product} ada fitur apa",
    "ada warna apa {product}",
]
```

#### Generic (no exclusive keywords)
```python
PRODUCT_DESCRIPTION_GENERIC = [
    # No "spec", "detail", "deskripsi"
    "{product} kayak gimana",
    "jelasin {product}",
    "{product} awet ga",
    "{product} bagus ga",
    "cocok ga {product}",
]
```

### F. OUT_OF_SCOPE Templates

```python
OUT_OF_SCOPE_PATTERNS = [
    # Refund/Return
    "mau refund",
    "cara refund",
    "bisa return",

    # Complaints
    "barang rusak",
    "mau komplain",
    "produk tidak sesuai",

    # Store info
    "jam buka toko",
    "alamat toko",

    # CONFUSING (looks like other intents)
    "harga refund berapa",  # has "harga berapa" but OOS
    "stok barang rusak",    # has "stok" but OOS
    "gimana cancel order",  # has "gimana order" but OOS
    "ready refund kapan",   # has "ready" but OOS
]

AMBIGUOUS_QUERIES = [
    # Genuinely unclear → label as out_of_scope
    "itu gimana",
    "yang itu",
    "berapa",
    "ada ga",
    "ready",
    "gimana",
    "yang kemarin",
    "terus gimana",
]
```

---

## Hard Negative Pairs (480 total)

### Category 1: Stock vs Payment (80 pairs)
```python
[
    ("ready stock {product}", "ready bayar"),
    ("{product} ready", "ready transfer"),
    ("ada stok {product}", "ada cicilan"),
    ("tersedia {product}", "tersedia {payment}"),
]
```

### Category 2: Price vs Stock (80 pairs)
```python
[
    ("berapa harga {product}", "berapa unit {product}"),
    ("{product} berapa", "stoknya berapa"),
    ("kena berapa", "tinggal berapa"),
]
```

### Category 3: Order vs Description (80 pairs)
```python
[
    ("gimana pesanan saya", "gimana {product}"),
    ("yang dipesan gimana", "yang ini gimana specnya"),
    ("info order {id}", "info {product}"),
]
```

### Category 4: Price vs Description (80 pairs)
```python
[
    ("info harga {product}", "info lengkap {product}"),
    ("detail harga", "detail {product}"),
    ("mau tau harga", "mau tau {product}"),
]
```

### Category 5: Stock vs Order (80 pairs)
```python
[
    ("restock kapan", "sampai kapan"),
    ("{product} ready kapan", "order ready kapan"),
    ("kapan ada lagi", "kapan sampai"),
]
```

### Category 6: All vs OOS (80 pairs)
```python
[
    ("harga {product}", "harga refund"),
    ("stok {product}", "stok barang rusak"),
    ("order saya gimana", "cancel order gimana"),
]
```

---

## Expected Training Curve

| Epoch | V3 Actual | V4 Target | What Model is Learning |
|-------|-----------|-----------|------------------------|
| 1 | 84% | 50-55% | Random + obvious patterns |
| 2 | 98% | 60-65% | Starting semantic learning |
| 3 | 99% | 70-75% | Disambiguating shared keywords |
| 4 | 99% | 78-82% | Handling hard negatives |
| 5 | 99% | 82-85% | Edge cases |
| 6-10 | 99% | 85-88% | Plateau (ceiling due to ambiguity) |

**Key Indicator**: If epoch 1 is still >70%, the dataset is still too easy.

---

## Implementation Checkpoints

### Checkpoint 1: Template Design (This document)
- [x] Define shared keyword mapping
- [x] Design keyword-based templates for each intent
- [x] Design generic templates for each intent
- [x] Define hard negative pairs
- [x] Define ambiguous queries

### Checkpoint 2: Data Generator Implementation
- [ ] Create `notebook/02c_data_generator_v4.ipynb`
- [ ] Implement shared keyword templates
- [ ] Implement generic templates
- [ ] Implement hard negative generator
- [ ] Implement ambiguous query generator
- [ ] Implement noise functions (same as V3)
- [ ] Implement deduplication
- [ ] Generate dataset

### Checkpoint 3: Dataset Validation
- [ ] Verify keyword distribution (no exclusive keywords)
- [ ] Verify data distribution (42/28/20/10%)
- [ ] Verify per-intent sample counts
- [ ] Sample review: manually check 50 random samples
- [ ] Save as `intent_dataset_v4.csv`

### Checkpoint 4: Training
- [ ] Create `notebook/03c_train_v4.ipynb`
- [ ] Train model on V4 data
- [ ] Record epoch-by-epoch accuracy
- [ ] Verify epoch 1 accuracy ~50-55%
- [ ] Verify final accuracy ~85-90%

### Checkpoint 5: Validation
- [ ] If accuracy >95%: Add more hard negatives
- [ ] If accuracy <80%: Reduce ambiguity
- [ ] Compare V3 vs V4 training curves
- [ ] Document findings

### Checkpoint 6: Thesis Documentation
- [ ] Compare V2/V3/V4 results
- [ ] Explain why lower accuracy is better
- [ ] Create visualizations
- [ ] Write analysis section

---

## Files to Create

| File | Status | Description |
|------|--------|-------------|
| `thoughts/shared/analysis-v3-still-overfitting.md` | Done | Analysis of why V3 still gets 98% |
| `thoughts/shared/plans/2025-12-05-v4-semantic-intent-classifier.md` | Done | This implementation plan |
| `notebook/02c_data_generator_v4.ipynb` | Todo | V4 data generator |
| `data/synthetic/intent_dataset_v4.csv` | Todo | V4 training data |
| `data/synthetic/intent_dataset_v4_test.csv` | Todo | V4 OOD test data |
| `notebook/03c_train_v4.ipynb` | Todo | V4 training notebook |

---

## Success Criteria

### Must Have
1. Epoch 1 accuracy: 50-60% (not 84%)
2. Final accuracy: 85-90% (not 98%)
3. No keyword appears in only 1 intent
4. 28% samples without intent-specific keywords

### Nice to Have
1. Training curve shows gradual improvement
2. Confidence scores correlate with correctness
3. Error analysis shows semantic confusion, not random errors

---

## Thesis Narrative

The thesis should present this progression:

1. **V2 (Baseline)**: Naive synthetic data generation
   - Result: 98.4% accuracy
   - Problem: Data leakage (same products in train/test)
   - Lesson: Proper train/test split is critical

2. **V3 (Fixed Leakage)**: Separate entity pools
   - Result: 98% accuracy (still high!)
   - Problem: Keywords exclusive per intent → keyword matching
   - Lesson: High accuracy can mask poor generalization

3. **V4 (Semantic Learning)**: Break keyword exclusivity
   - Result: 85-90% accuracy
   - Validation: Gradual training curve proves semantic learning
   - Lesson: Synthetic data requires intentional ambiguity

4. **Conclusion**: Lower accuracy on synthetic data is BETTER if it indicates:
   - True semantic understanding
   - Better generalization to real-world queries
   - More robust model for production use
