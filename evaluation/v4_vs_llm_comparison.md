# V4 vs LLM Comparison Results

Generated: 2025-12-06T02:53:14.075722


| Feature              | V4 Classifier    | LLM Slot Extraction |
|----------------------|------------------|---------------------|
| Model                | DistilBERT (66M) | Qwen3-8B (8B)       |
| Model Size           | ~250MB           | ~5GB (Q4_K_M)       |
| Training Data        | 2,400 samples    | Zero (prompt only)  |
| Task Accuracy        | 78.8%            | 95.7% (on valid)    |
| Response Rate        | 100%             | 57.5%              |
| Latency              | 43ms             | 4.30s              |
| Multi-intent         | No               | Yes (100%)          |
| Entity Extraction    | No               | Yes (88%)           |
| Clarification        | No               | Yes (98%)           |
| Flexibility          | Fixed 6 intents  | Unlimited via prompt|


## Summary

- V4 Accuracy: 78.8%
- LLM Accuracy (valid): 95.7%
- LLM Error Rate: 42.5%
- Both Correct: 51.2%
- Speed Difference: V4 is 99x faster
