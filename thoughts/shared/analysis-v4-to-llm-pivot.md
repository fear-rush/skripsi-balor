# Analysis: From Intent Classification to LLM Slot Extraction

**Date**: 2025-12-05
**Context**: After V4 training results, pivoting thesis direction to modern LLM approach

---

## Executive Summary

This document captures our complete journey from traditional intent classification (V2→V3→V4) to the decision to pivot to LLM-based semantic slot extraction. Key findings:

1. **V4 partially succeeded**: Epoch 1 accuracy dropped from 84% to 44%, proving we broke keyword exclusivity
2. **But transformers are too powerful**: Model still achieves 98% by finding other patterns in template-based data
3. **Synthetic data has a ceiling**: Any template-based dataset will eventually be "solved"
4. **Modern approach is better**: LLM-based slot extraction is the 2024-2025 industry standard

---

## Part 1: Intent Classification Journey (V2 → V3 → V4)

### Version Progression

| Version | Problem Addressed | Epoch 1 | Final | Root Cause |
|---------|-------------------|---------|-------|------------|
| V2 | None | ~85% | 98.4% | Data leakage (same products in train/test) |
| V3 | Data leakage | 84% | 98% | Keywords exclusive per intent |
| V4 | Keyword exclusivity | **44%** | 98% | Model finds template patterns |

### V4 Training Results

```
Epoch 1:  44.4% (1.60 loss)  ✅ Target: 50-55%
Epoch 2:  91.8% (0.80 loss)  ❌ Jumped too fast
Epoch 3:  96.6% (0.21 loss)
Final:    97.9% (0.004 loss) ❌ Same as V3
```

### Key Insight

**Epoch 1 success proves we broke keyword exclusivity.** The model no longer does:
- "harga" → product_price (99%)
- "stok" → product_stock (99%)

But by Epoch 2, DistilBERT (66M parameters) finds OTHER patterns:
- Template structures
- Word order patterns
- Sentence length distributions
- Semantic clusters ("mahal/worth it" signal price)

### Why V4 Still Achieves 98%

1. **Transformer models are too powerful**
   - Even "generic" templates have predictable structures
   - Model learns meta-patterns we didn't intend

2. **Template-based data has inherent patterns**
   ```
   product_price: "{product} mahal ga" → product + adjective
   product_stock: "masih bisa dibeli" → modal + verb
   ```

3. **Ambiguous queries don't hurt accuracy**
   - 10% labeled as out_of_scope
   - Correctly classified → HELPS accuracy

---

## Part 2: The Fundamental Limitation of Synthetic Data

### What Synthetic Data CAN Test
- Pattern recognition capability
- Keyword-to-intent mapping
- Basic context understanding
- Known edge cases

### What Synthetic Data CANNOT Test
- Real distributional shift
- Novel expressions
- Implicit conversation context
- Pragmatic ambiguity
- Multi-intent queries
- Confidence calibration

### The 98% Paradox

```
Synthetic accuracy ≈ 98%  →  Real-world accuracy ≈ ???

The gap could be:
- 98% synthetic → 90% real (good)
- 98% synthetic → 70% real (poor)
- 98% synthetic → 50% real (catastrophic)

We cannot know without real data.
```

---

## Part 3: Modern Approach - LLM Semantic Slot Extraction

### Industry Standard (2024-2025)

Major companies now use LLM → JSON instead of intent classification:
- Shopify AI
- LINE AI
- Traveloka internal AI
- Tokopedia new bot

### Example

```
User: "cek pesanan saya 99871, yang tadi error"

LLM Output:
{
  "task": "check_order_status",
  "order_id": "99871",
  "urgency": "normal",
  "error_related": true
}
```

### Comparison

| Aspect | Intent Classification | LLM Slot Extraction |
|--------|----------------------|---------------------|
| Training Data | 2,400+ samples | Zero (prompt only) |
| Flexibility | Fixed 6 intents | Unlimited tasks |
| Multi-intent | Not supported | Natural |
| Indonesian | Needs fine-tuned model | Multilingual |
| Latency | ~10ms | ~200-500ms |
| Cost | Free | Free (self-hosted) |
| Maintenance | Retrain | Update prompt |

---

## Part 4: Self-Hosted LLM Options for Indonesian

### Top Recommendations

| Rank | Model | Size | Indonesian | JSON | VRAM |
|------|-------|------|------------|------|------|
| 1 | **Qwen3-8B** | 8B | Multilingual (119 langs) | Excellent | 4-8GB |
| 2 | **Sahabat-AI 8B** | 8B | Native Indonesian | Good | 8-16GB |
| 3 | **Komodo-7B** | 7B | Native + 11 regional | Good | 3.5-7GB |
| 4 | **Cendol** | 300M-13B | Native Indonesian | Good | 2-13GB |
| 5 | **SeaLLM-13B** | 13B | SEA multilingual | Good | 13GB |

### Recommended: Qwen3-8B

- 119 language support including Indonesian
- Best JSON structured output (native support)
- Easy deployment: `ollama run qwen3:8b`
- Apache 2.0 license - fully commercial
- 4-8GB VRAM - runs on consumer GPU

### Alternative: Sahabat-AI 8B

- Native Indonesian model by GoTo + Indosat
- Supports regional languages (Javanese, Sundanese, etc.)
- Better for informal Indonesian ("gue", "lo", "ga")

### Hardware Requirements

| Budget | GPU | Models Supported |
|--------|-----|------------------|
| Entry | RTX 4060 Ti 16GB (~$400) | All 7-8B quantized |
| Mid | RTX 4070 Ti SUPER (~$800) | All 8-13B |
| High | RTX 4090 24GB (~$1600) | Everything |

---

## Part 5: Decision and Path Forward

### Final Decision

1. **Thesis Structure**: Include V2→V3→V4 as comparison chapter
2. **Main Focus**: LLM-based slot extraction with self-hosted Qwen3-8B
3. **Production**: Self-hosted (no API costs)

### Thesis Outline

**Part 1: Traditional Approach (Background)**
- Chapter 2.x: Intent Classification with DistilBERT
  - V2: Data leakage → 98.4%
  - V3: Fixed leakage → 98%
  - V4: Shared keywords → 98%
  - Conclusion: Synthetic data ceiling

**Part 2: Modern Approach (Main Contribution)**
- Chapter 3: LLM-Based Semantic Slot Extraction
  - Self-hosted Qwen3-8B + Ollama
  - Prompt engineering for Indonesian
  - Multi-intent and entity extraction
  - Bagisto integration

**Part 3: Evaluation**
- Chapter 4: Comparison
  - Intent classifier vs LLM extraction
  - Accuracy, latency, cost analysis

---

## Part 6: Implementation Plan

### Architecture

```
User Query → LLM (Qwen3-8B) → JSON Output → Route to API
                ↓
    {
      "task": "check_stock",
      "entities": {"product_name": "jaket biru"},
      "multi_intent": ["ask_price"],
      "confidence": 0.92
    }
                ↓
    Bagisto API → Response
```

### Setup Steps

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen3:8b
   ```

2. Slot Extraction:
   ```python
   import ollama

   def extract_slots(query: str) -> dict:
       response = ollama.chat(
           model='qwen3:8b',
           messages=[
               {'role': 'system', 'content': SYSTEM_PROMPT},
               {'role': 'user', 'content': query}
           ],
           format='json'
       )
       return json.loads(response['message']['content'])
   ```

### Files to Create

| File | Description |
|------|-------------|
| `notebook/04_llm_slot_extraction.ipynb` | LLM implementation |
| `src/chatbot/llm_extractor.py` | Production code |
| `data/evaluation/real_queries.json` | Test dataset |

---

## Part 7: Key Takeaways

### What We Learned

1. **Breaking keyword exclusivity works** (44% Epoch 1 vs 84%)
2. **But transformers find other patterns** in template data
3. **Synthetic data has a ceiling** regardless of design
4. **Modern LLM approach is superior** for production

### For the Thesis

The V2→V3→V4 progression demonstrates:
- Importance of proper data engineering
- Limitations of synthetic data
- Why industry moved to LLM-based approaches

### Research Contribution

1. Methodology for creating challenging synthetic intent data
2. Analysis of synthetic data limitations for NLP
3. Comparison of traditional vs modern chatbot NLU
4. Self-hosted LLM solution for Indonesian e-commerce

---

## References

### Self-Hosted LLMs
- [Sahabat-AI](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Komodo-7B](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base)
- [Cendol](https://huggingface.co/indonlp/cendol)

### Techniques Researched (Not Implemented)
- Label Smoothing
- Focal Loss
- Contrastive Learning
- Curriculum Learning
- R-Drop Regularization

### Academic Papers
- "Learning label smoothing for text classification" (PeerJ, 2024)
- "Multi-Granularity Hard-Negative Synthesis" (ArXiv, 2024)
- "Few-shot intent detection with contrastive learning" (ScienceDirect, 2024)
- "Study of Text Augmentation with Paraphrasing for Indonesian" (2024)
