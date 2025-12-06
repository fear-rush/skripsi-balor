# Analysis: Why 98.44% Accuracy is "Too Good to Be True"

**Date**: 2025-12-05
**Author**: Claude (Analysis for Firas's Thesis)

## Executive Summary

The intent classifier achieved **98.44% test accuracy**, which is suspiciously high for a real-world NLP task. After thorough analysis of the data generation and training pipeline, I've identified **6 critical issues** that explain this artificially inflated performance.

**Bottom line**: The model has learned to recognize patterns in the synthetic data, not to understand user intent. It will likely fail dramatically on real customer queries.

---

## Critical Issues Identified

### Issue 1: Severe Template Reuse (CRITICAL)

**Problem**: Only **151 base templates** generate **3,000 samples**. Each template appears ~20 times with minor variations.

```
Templates per intent:
- order_status: 27 patterns
- payment_info: 23 patterns
- product_price: 20 patterns
- product_stock: 20 patterns
- product_description: 21 patterns
- out_of_scope: 40 patterns
```

**Impact**: The model memorizes template structure, not semantic meaning.

**Example** - All these are essentially the same pattern:
```
"harga {product} berapa"
"berapa harga {product}"
"{product} harganya berapa"
"price {product}"
```

### Issue 2: High Duplication Rate (CRITICAL)

**Problem**: **262 duplicate samples** (8.7% of dataset), heavily concentrated in specific intents:

| Intent | Duplication Rate |
|--------|-----------------|
| payment_info | 30% |
| out_of_scope | 30% |
| order_status | 10% |
| product_stock | 0.7% |
| product_description | 0.6% |
| product_price | 0.2% |

**Most repeated samples** (appearing 10-11 times each):
- "permisi kak untuk pembayaran tersedia metode apa saja"
- "metode byr apa"
- "orderan saya gimana"
- "bayar pake apa sih"

**Impact**: If train/test split doesn't deduplicate first, the **same exact text appears in both sets** = data leakage.

### Issue 3: Same Entity Pool for Train & Test (CRITICAL)

**Problem**: All 3,000 samples draw from the **same finite pools**:
- 93 products (from Bagisto database)
- 18 order ID formats
- 27 payment methods

**Impact**: No **out-of-distribution testing**. The model never sees:
- Products it wasn't trained on
- Order ID formats it hasn't memorized
- Payment methods with different names

**Example of leakage**:
```
Training: "harga arctic bliss winter scarf berapa"
Test:     "harga arctic cozy knit beanie berapa"  # Same pool, same pattern
```

### Issue 4: Superficial Noise (Not Semantic Ambiguity)

**Problem**: The noise injection creates **character-level** variations, not **semantic ambiguity**:

```python
# Current noise: character substitution
"harga" → "hrg" → "h4rg4" → "@hrg"

# Missing: semantic ambiguity
"stok ready ga?" → Could mean: stock availability OR payment readiness
"gimana produknya?" → Could mean: description OR price OR availability
```

**Noise distribution**:
- 70% of samples are "low" or "medium" noise (barely different from template)
- Only 10% are "extreme" noise

**Impact**: Model learns that typos don't matter, but never learns to handle genuinely ambiguous queries.

### Issue 5: No Hard Negatives

**Problem**: Each intent has **clearly distinct vocabulary**. No samples challenge the model with:

| Missing Challenge | Example |
|------------------|---------|
| Similar queries, different intents | "bisa stok?" (stock) vs "bisa bayar?" (payment) |
| Vague queries | "gimana ini" (could be anything) |
| Multi-intent queries | "harga dan stok beanie" (price + stock) |
| Sarcasm/negation | "harganya mahal banget, stok kosong pula" |

**Impact**: Model can achieve high accuracy with simple **keyword matching**:
- "harga", "berapa" → product_price
- "stok", "ready" → product_stock
- "bayar", "transfer" → payment_info

### Issue 6: Out-of-Scope Too Distinct

**Problem**: The 40 out-of-scope patterns are **obviously unrelated** to e-commerce:

```python
"siapa presiden indonesia"
"bagaimana cuaca hari ini"
"ceritakan tentang AI"
"mau kerja disini"
```

**Impact**: Real out-of-scope queries from customers would be much harder:
- "mau komplain nih"
- "kemarin salah kirim"
- "barangnya beda sama foto"

These look like product/order queries but require human handling.

---

## Evidence of Overfitting

### 1. Training Stopped at Epoch 4 (Early Stopping)

```
Epoch 1: F1 = 0.984410 (BEST)
Epoch 2: F1 = 0.984376
Epoch 3: F1 = 0.982218
Epoch 4: F1 = 0.984348
```

The model **peaked at epoch 1** and couldn't improve. This suggests:
- Data is too easy - model learns everything in first epoch
- No room for generalization - just memorization

### 2. Suspiciously Uniform Accuracy Across Intents

If test accuracy is ~98% across all 6 intents, it suggests:
- No genuinely hard categories
- All intents are equally "solvable" by pattern matching

### 3. Confidence Scores Too High

If average confidence is >0.95 even for correct predictions, the model is **overconfident** - a classic sign of training on easy data.

---

## What Would Happen with Real Data?

Predicted failure modes when deployed:

| Real Query | Expected Failure |
|------------|------------------|
| "barang blm nyampe 3 hari" | Classified as order_status, but actually needs escalation |
| "sepatu nike size 42" | Fails - product not in training set |
| "bs tf bca g" | May work (seen similar) |
| "mau cancel dong" | Misclassified - not in out_of_scope training |
| "harga sama stoknya brp" | Fails - multi-intent not supported |
| "produknya bagus ga sih" | Wrong - sounds like description but is review request |

---

## Recommendations for Improvement

### Short-term Fixes

1. **Deduplicate before splitting**: Remove all duplicate texts, then split train/test
2. **Stratified split by noise level**: Ensure each noise level appears proportionally
3. **Hold-out products**: Reserve 20% of products for test-only
4. **Add hard negatives**: Create confusing pairs that test semantic understanding

### Medium-term Improvements

1. **Increase template diversity**: Aim for 500+ unique templates (not 151)
2. **Add semantic noise**: Paraphrase templates using LLM, not just typos
3. **Collect real data**: Even 100 real customer queries would be more valuable
4. **Cross-validate**: Use k-fold CV to detect overfitting

### Long-term Strategy

1. **Active learning**: Deploy with low confidence threshold, collect failures
2. **Human-in-the-loop**: Have humans label edge cases
3. **Adversarial testing**: Systematically try to confuse the model
4. **Real-world evaluation**: Test on actual customer support logs (anonymized)

---

## Proposed Experiment: Honest Evaluation

To get a realistic accuracy estimate, I recommend:

### Experiment 1: Out-of-Distribution Products
- Train on products 1-70
- Test on products 71-93 (never seen)
- Expected accuracy drop: 20-40%

### Experiment 2: Template Holdout
- Train on 80% of templates
- Test only on held-out 20% templates
- Expected accuracy drop: 30-50%

### Experiment 3: Real Query Test
- Collect 50 real customer queries (from logs or manual creation)
- Test without any fine-tuning
- Expected accuracy: 40-60%

---

## Conclusion

The **98.44% accuracy is not meaningful** because:

1. Train and test data are too similar (same templates, same products)
2. High duplication causes data leakage
3. Noise is cosmetic, not semantic
4. No genuinely challenging samples

**Realistic expectation**: On real customer data, this model would likely achieve **50-70% accuracy** - a significant gap that the thesis should acknowledge and address.

The thesis should present this as a **baseline** that demonstrates the limitations of synthetic data, then show improvements with the recommended strategies above.
