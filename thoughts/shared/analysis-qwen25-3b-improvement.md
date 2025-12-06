# Qwen2.5-3B vs Qwen3-8B: Model Improvement Analysis

**Date**: 2025-12-06
**Evaluation**: 80 Indonesian e-commerce queries
**Previous Model**: Qwen3-8B-GGUF Q4_K_M (8B params, ~5GB)
**Current Model**: Qwen2.5-3B-Instruct Q8_0 (3B params, ~3.6GB)

---

## 1. Executive Summary

### The GPU Memory Problem: SOLVED

The switch from Qwen3-8B to Qwen2.5-3B eliminated the GPU memory saturation issue completely:

| Metric | Qwen3-8B (Q4_K_M) | Qwen2.5-3B (Q8_0) | Change |
|--------|-------------------|-------------------|--------|
| Parameters | 8B | 3B | -62.5% |
| Model Size | ~5.0 GB | ~3.6 GB | -28% |
| Quantization | Q4 (4-bit) | Q8 (8-bit) | Higher quality |
| **Error Rate** | 42.5% | **0%** | **-42.5%** |
| **Response Rate** | 57.5% | **100%** | **+42.5%** |
| Overall Accuracy | 55.0% | **90.0%** | **+35%** |
| Accuracy on Valid | 95.7% | 90.0% | -5.7% |
| Mean Latency | 4.3s | 2.6s | -40% |

### Final Verdict: LLM Definitively Beats V4

| Model | Accuracy | Response Rate | Latency |
|-------|----------|---------------|---------|
| V4 (DistilBERT) | 78.8% | 100% | 46ms |
| **LLM (Qwen2.5-3B)** | **90.0%** | 100% | 2.6s |

**Winner: LLM by 11.2 percentage points**

---

## 2. Why the Smaller Model Works Better

### 2.1 Memory Analysis

The Qwen3-8B model caused GPU memory saturation on M1 Pro (16GB unified memory):

| Component | Qwen3-8B | Qwen2.5-3B |
|-----------|----------|------------|
| Model Weights | ~5GB | ~3.6GB |
| KV Cache (4096 ctx) | ~1-2GB | ~0.5-1GB |
| System Overhead | ~2-3GB | ~2-3GB |
| **Total** | ~10-12GB | ~7-8GB |
| **Headroom** | Only 4-6GB | **8-9GB** |

With Qwen2.5-3B:
- Sufficient headroom prevents memory pressure
- Stable KV cache allocation
- No query failures due to OOM

### 2.2 Quantization Quality Trade-off

| Quantization | Bits | Precision Loss | Model Size |
|--------------|------|----------------|------------|
| Q4_K_M | 4-bit | Moderate | Smaller |
| **Q8_0** | 8-bit | **Minimal** | Larger per param |

The Q8 quantization on a 3B model provides:
- **Better numerical precision** than Q4 on 8B
- Fewer inference errors from weight quantization
- More stable generation behavior

### 2.3 Latency Improvement

| Model | Mean Latency | Min | Max | Stability |
|-------|--------------|-----|-----|-----------|
| Qwen3-8B | 4.3s | 0.5s (fail) | 11.0s | Bimodal |
| **Qwen2.5-3B** | **2.6s** | 2.2s | 2.9s | **Stable** |

The bimodal latency distribution of Qwen3-8B (0.5s failures vs 4-11s successes) is eliminated. Qwen2.5-3B shows consistent ~2.6s latency across all queries.

---

## 3. Accuracy Trade-off Analysis

### 3.1 The Real Trade-off

We traded 5.7% accuracy on valid responses for 42.5% reliability improvement:

| Metric | Qwen3-8B | Qwen2.5-3B |
|--------|----------|------------|
| Accuracy on Valid Responses | 95.7% | 90.0% |
| Total Correct Answers | 44/80 | **72/80** |
| Queries Actually Answered | 46/80 | **80/80** |

**Net Result: +63.6% more correct answers** (72 vs 44)

### 3.2 Why This Trade-off is Worth It

1. **For production chatbots**, a response is always better than silence
2. **User experience**: 90% accuracy with 100% response rate beats 95.7% accuracy with 57.5% response rate
3. **Reliability is non-negotiable** for customer-facing applications

---

## 4. Head-to-Head Comparison

### 4.1 Agreement Matrix

| Outcome | Count | Percentage | Interpretation |
|---------|-------|------------|----------------|
| Both Correct | 58 | 72.5% | Models agree correctly |
| Both Wrong | 3 | 3.8% | Hard queries for both |
| V4 Only Correct | 5 | 6.2% | V4 advantages |
| **LLM Only Correct** | **14** | **17.5%** | **LLM advantages** |

LLM beats V4 on 14 queries where V4 failed - that's **2.8x** more exclusive wins.

### 4.2 Category-Level Performance

| Category | Total | V4 Acc | LLM Acc | Diff | Winner |
|----------|-------|--------|---------|------|--------|
| check_stock | 15 | 80.0% | 86.7% | +6.7% | LLM |
| ask_price | 15 | 93.3% | **100%** | +6.7% | **LLM** |
| check_order | 12 | 100% | 100% | 0% | Tie |
| ask_payment | 10 | 100% | 100% | 0% | Tie |
| product_info | 10 | 40.0% | **90%** | **+50%** | **LLM** |
| out_of_scope | 8 | **87.5%** | 75.0% | -12.5% | V4 |
| multi_intent | 10 | 40.0% | **70%** | **+30%** | **LLM** |

### Key Insights:

1. **LLM dominates product_info**: +50% improvement (90% vs 40%)
   - V4 confused "berapa" questions (e.g., "RAM berapa GB?") as price queries
   - LLM correctly identifies spec/feature questions

2. **LLM dominates multi_intent**: +30% improvement (70% vs 40%)
   - V4 can only output one intent
   - LLM extracts primary task + secondary intents

3. **V4 better at out_of_scope**: +12.5% advantage
   - LLM sometimes over-interprets vague queries
   - V4 has explicit out_of_scope training

4. **Perfect tie on check_order & ask_payment**: Both achieve 100%
   - Clear patterns in training data
   - Unambiguous user intent

---

## 5. Detailed Query Analysis

### 5.1 LLM Exclusive Wins (14 queries)

Queries where LLM succeeded but V4 failed:

| Query | Expected | V4 | LLM | LLM Advantage |
|-------|----------|-----|-----|---------------|
| "kaos polos L ada?" | check_stock | out_of_scope | check_stock | Short/informal |
| "kemeja putih M kosong ga?" | check_stock | out_of_scope | check_stock | "kosong" = out of stock |
| "kacamata hitam rdy ga?" | check_stock | out_of_scope | check_stock | "rdy" slang |
| "ongkir ke jakarta berapa?" | ask_price | out_of_scope | ask_price | Shipping = price |
| "warna apa aja yang tersedia?" | product_info | product_stock | product_info | Color = info, not stock |
| "hp ini RAM berapa GB?" | product_info | product_price | product_info | Spec question |
| "garansi produk ini berapa lama?" | product_info | out_of_scope | product_info | Warranty = info |
| "sepatu ini waterproof ga?" | product_info | product_price | product_info | Feature question |
| "ada size chart ga kak?" | product_info | product_price | product_info | Size chart = info |
| "kontak customer service berapa?" | out_of_scope | payment_info | out_of_scope | CS != payment |
| "stok hp samsung ada ga? kalo ada harga berapa?" | check_stock | out_of_scope | check_stock | Multi-intent |
| "bisa COD ga? stok tas nya ada?" | ask_payment | product_stock | ask_payment | Multi-intent primary |
| "ready stok kaos? warna apa aja? harga berapa?" | check_stock | out_of_scope | check_stock | Triple-intent |
| "sepatu ini bahan apa? waterproof ga? harga berapa?" | product_info | product_price | product_info | Multi-intent |

**Pattern Analysis:**
- 6/14 are **multi-intent queries** (V4 cannot handle)
- 4/14 are **product_info misclassified** as price by V4
- 3/14 are **short/informal stock queries** V4 missed
- 1/14 is **out_of_scope edge case**

### 5.2 V4 Exclusive Wins (5 queries)

| Query | Expected | V4 | LLM | V4 Advantage |
|-------|----------|-----|-----|--------------|
| "masih bisa beli hp samsung ga?" | check_stock | product_stock | ask_price | "beli" interpretation |
| "headset bluetooth ready?" | check_stock | product_stock | product_info | "ready" = stock |
| "bisa tukar ukuran ga?" | out_of_scope | out_of_scope | ask_payment | Exchange = OOS |
| "mau cancel pesanan gimana?" | out_of_scope | out_of_scope | check_order | Cancel = OOS |
| "mau beli hp, ada yang RAM 8GB ga?" | check_stock | product_stock | product_info | Stock check |

**Pattern Analysis:**
- 2/5 are **out_of_scope edge cases** (exchange, cancel)
- 3/5 are **stock queries** LLM over-interpreted as info/price

### 5.3 Both Wrong (3 queries)

| Query | Expected | V4 | LLM | Issue |
|-------|----------|-----|-----|-------|
| "layar laptop ini berapa inch?" | product_info | product_price | ask_price | Both fooled by "berapa" |
| "jaket ini ready ga? gimana cara bayarnya?" | check_stock | payment_info | check_order | Multi-intent confusion |
| "laptop asus specs nya gimana? harganya berapa?" | product_info | product_price | ask_price | Multi-intent priority |

---

## 6. LLM Additional Capabilities

Beyond classification accuracy, LLM provides capabilities V4 cannot match:

### 6.1 Entity Extraction

**Query**: "ready stock sepatu nike size 42?"

| Model | Output |
|-------|--------|
| V4 | `{"intent": "product_stock", "confidence": 0.976}` |
| LLM | `{"task": "check_stock", "entities": {"product_name": "sepatu nike", "variant": "size 42"}}` |

LLM extracted **product name** AND **size variant**. This enables:
- Direct database queries
- Personalized responses
- Better search results

### 6.2 Multi-Intent Detection

**Query**: "harga sepatu nike berapa? ada size 43 ga?"

| Model | Output |
|-------|--------|
| V4 | `{"intent": "product_price"}` |
| LLM | `{"task": "ask_price", "multi_intent": ["check_stock"], "entities": {"product_name": "sepatu nike", "variant": "size 43"}}` |

LLM identified **both intents** and the variant. V4 can only pick one.

### 6.3 Payment Method Extraction

**Query**: "bisa bayar pake gopay ga?"

| Model | Output |
|-------|--------|
| V4 | `{"intent": "payment_info"}` |
| LLM | `{"task": "ask_payment", "entities": {"payment_method": "gopay"}}` |

LLM extracted the specific payment method mentioned.

### 6.4 Order ID Extraction

**Query**: "pesanan saya 12345 udah sampai mana?"

| Model | Output |
|-------|--------|
| V4 | `{"intent": "order_status"}` |
| LLM | `{"task": "check_order", "entities": {"order_id": "12345"}}` |

LLM extracted the order ID for direct lookup.

---

## 7. Comparison with Previous Qwen3-8B Analysis

### 7.1 Before vs After

| Aspect | Qwen3-8B Analysis | Qwen2.5-3B Analysis |
|--------|-------------------|---------------------|
| Headline Result | "LLM 55% vs V4 78.8%" | **"LLM 90% vs V4 78.8%"** |
| Fair Comparison? | No (42.5% failures) | **Yes (0% failures)** |
| Recommended Approach | "Use valid responses only" | **"Use overall accuracy"** |
| Thesis Narrative | "LLM has potential but hardware limited" | **"LLM definitively better"** |

### 7.2 What Changed

1. **Reliability**: 0% errors vs 42.5% errors
2. **Fair comparison**: Now possible
3. **Clear winner**: LLM by 11.2 points
4. **Practical deployment**: Consumer hardware viable

---

## 8. Thesis Implications

### 8.1 Recommended Narrative

> "We evaluated two approaches for Indonesian e-commerce chatbot NLU: a fine-tuned DistilBERT intent classifier (V4) and LLM-based semantic slot extraction using Qwen2.5-3B-Instruct.
>
> On 80 test queries, the LLM achieved **90% accuracy** compared to V4's 78.8%, while providing additional capabilities: entity extraction, multi-intent detection, and contextual understanding of informal Indonesian.
>
> The LLM demonstrated particular strength on:
> - **Product information queries**: 90% vs 40% (+50%)
> - **Multi-intent queries**: 70% vs 40% (+30%)
> - **Informal/slang queries**: Better handling of "rdy", "ga", "gimana"
>
> With 100% response rate and 2.6s average latency, the 3B parameter model proves viable for production deployment on consumer hardware (Apple M1 Pro, 16GB unified memory)."

### 8.2 Key Points for Thesis

1. **LLM beats traditional classifier**: 90% vs 78.8% (+11.2%)
2. **100% reliability**: Zero failed queries
3. **Rich output**: Entity extraction, multi-intent detection
4. **Practical deployment**: Works on consumer hardware
5. **Category strengths**: product_info (+50%), multi_intent (+30%)
6. **Trade-off acknowledged**: Slightly higher latency (2.6s vs 46ms)

### 8.3 Limitations to Note

1. **Latency**: 56x slower than V4 (2.6s vs 46ms)
2. **Resource requirements**: Requires GPU, ~8GB memory
3. **out_of_scope handling**: V4 slightly better (87.5% vs 75%)
4. **Edge cases**: Some stock queries misclassified as product_info

---

## 9. Conclusion

The switch from Qwen3-8B (Q4_K_M) to Qwen2.5-3B (Q8_0) resolved the GPU memory saturation problem and enabled a **fair, definitive comparison** between LLM and traditional intent classification.

### Final Results

| Approach | Accuracy | Response Rate | Extra Capabilities |
|----------|----------|---------------|-------------------|
| V4 (DistilBERT) | 78.8% | 100% | None |
| **LLM (Qwen2.5-3B)** | **90.0%** | 100% | Entity extraction, multi-intent |

**Verdict**: LLM-based semantic slot extraction is the superior approach for Indonesian e-commerce chatbots, achieving higher accuracy while providing richer structured output for downstream processing.

---

## 10. Files Reference

| File | Description |
|------|-------------|
| `evaluation/llm_slot_qwen2.json` | Current comparison results |
| `evaluation/v4_vs_llm_comparison.json` | Previous Qwen3-8B results |
| `thoughts/shared/analysis-v4-vs-llm-comparison.md` | Previous analysis |
| `src/llm_slot_extractor.py` | LLM extractor module |
| `scripts/compare_v4_vs_llm.py` | Comparison script |
| `models/llm/qwen2.5-3b-instruct-q8_0.gguf` | Current model file |
