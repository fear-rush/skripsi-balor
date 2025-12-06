# V4 vs LLM Comparison Results

Generated: 2025-12-06T03:53:54.506550


## Comparison Table: V4 vs LLM

| Metric | V4 Intent Classifier | LLM Slot Extraction |
|--------|---------------------|---------------------|
| Model | DistilBERT (66M params) | Qwen2.5-3B (3B params) |
| Model Size | ~250MB | ~3.6GB (Q8_0) |
| Training Data | 2,400 samples | Zero (prompt only) |
| Task Accuracy | 78.8% | 90.0% (on valid) |
| Response Rate | 100% | 100.0% |
| Latency | 46ms | 2.61s |
| Multi-intent | Not supported | 100% |
| Entity Extraction | Not supported | 88.2% |
| Clarification | Not supported | 98% |


## Summary

- V4 Accuracy: 78.8%
- LLM Accuracy (valid): 90.0%
- LLM Error Rate: 0.0%
- Both Correct: 72.5%
