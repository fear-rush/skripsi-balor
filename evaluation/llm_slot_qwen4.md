# V4 vs LLM Comparison Results

Generated: 2025-12-06T04:19:05.980440


## Comparison Table: V4 vs LLM

| Metric | V4 Intent Classifier | LLM Slot Extraction |
|--------|---------------------|---------------------|
| Model | DistilBERT (66M params) | Qwen3-4B-Instruct-2507 (4B params) |
| Model Size | ~250MB | ~2.6GB (Q4_K_M) |
| Training Data | 2,400 samples | Zero (prompt only) |
| Task Accuracy | 78.8% | 96.2% (on valid) |
| Response Rate | 100% | 100.0% |
| Latency | 41ms | 2.26s |
| Multi-intent | Not supported | 100% |
| Entity Extraction | Not supported | 88.2% |
| Clarification | Not supported | 98% |


## Summary

- V4 Accuracy: 78.8%
- LLM Accuracy (valid): 96.2%
- LLM Error Rate: 0.0%
- Both Correct: 77.5%
