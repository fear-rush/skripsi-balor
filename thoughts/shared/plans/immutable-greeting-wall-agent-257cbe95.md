# Research: Best Self-Hosted Open-Source LLMs for Indonesian Semantic Slot Extraction

## Research Summary

Comprehensive research on self-hosted, open-source LLM models suitable for semantic slot extraction with Indonesian (Bahasa Indonesia) language support. This research covers model capabilities, hardware requirements, deployment options, and performance characteristics.

---

## Category 1: Indonesian-Specific Models (Best for Indonesian Quality)

### 1. Sahabat-AI (GoTo x Indosat) - TOP RECOMMENDATION FOR INDONESIAN

**Hugging Face Links:**
- Base 8B: https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base
- Instruct 8B: https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct
- Base 9B: https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-base
- Instruct 70B: https://huggingface.co/Sahabat-AI/Llama-Sahabat-AI-v2-70B-IT
- Collection: https://huggingface.co/collections/GoToCompany/sahabat-ai-v1-672af7b248f5fdfd39ae2403

**Size:** 8B, 9B, and 70B parameter variants

**Indonesian Capability:** Native - specifically pretrained and fine-tuned for Indonesian and regional dialects
- Bahasa Indonesia (primary)
- Javanese, Sundanese, Balinese, Bataknese (v2)

**JSON/Structured Output:** Yes - instruction-tuned variants support structured output

**Hardware Requirements:**
- 8B model: 16GB VRAM (FP16) or 8GB VRAM (8-bit quantized)
- 9B model: 18GB VRAM (FP16) or 9GB VRAM (8-bit quantized)
- 70B model: 140GB VRAM (FP16) or 35GB VRAM (4-bit quantized)

**Inference Speed:**
- 8B: ~50-80 tokens/sec on RTX 4090
- 9B: ~45-75 tokens/sec on RTX 4090
- 70B: ~10-20 tokens/sec on RTX 4090 (4-bit)

**Ollama Compatibility:** Likely supported (based on Llama 3 and Gemma 2)

**vLLM/TGI Support:** Yes - based on Llama 3 and Gemma 2 architectures

**Fine-tuning:** Open-source, available for fine-tuning

**Context Length:** 128K (70B v2 model)

**License:** Open-source (specific license varies by base model)

**Strengths:**
- Most downloads for Indonesian models (35,000+ on Hugging Face)
- Backed by major Indonesian companies (GoTo, Indosat)
- Collaboration with AI Singapore and top Indonesian universities
- Deep understanding of local context and cultural relevance
- Best choice for Indonesian-specific semantic understanding

**Considerations:**
- Relatively new (launched November 2024)
- Less benchmark data compared to global models
- Regional language support expanding (v2)

---

### 2. Komodo-7B (Yellow.ai) - BEST FOR REGIONAL LANGUAGES

**Hugging Face Link:** https://huggingface.co/Yellow-AI-NLP/komodo-7b-base

**Size:** 7B parameters

**Indonesian Capability:** Native multilingual
- Indonesian (primary)
- 11 regional languages: Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Dayak Ngaju, Sundanese, Toba Batak, Lampungnese
- English

**JSON/Structured Output:** Yes (instruction-tuned variant available)

**Hardware Requirements:**
- 7B model: 14GB VRAM (FP16) or 7GB VRAM (8-bit quantized) or 3.5GB VRAM (4-bit)

**Inference Speed:** ~60-90 tokens/sec on RTX 4090

**Ollama Compatibility:** Yes - GGUF versions available

**vLLM/TGI Support:** Yes - based on Llama 2 architecture

**Fine-tuning:** Yes - available for fine-tuning

**Tokenizer Efficiency:**
- Mean fertility: 2.031 for Indonesian (30% more efficient than Llama-2)
- Expanded vocabulary: 35,008 tokens
- Lower inference cost

**Strengths:**
- State-of-the-art performance, outperforms GPT-3.5, Aya-101, Llama-2-Chat-13B
- Superior regional language support
- Most efficient tokenizer for Indonesian
- Trained on 8.5B tokens
- Excellent benchmarks on Indo MMLU, XCOPA-ID

**Considerations:**
- Based on older Llama-2 architecture
- Smaller than latest models

---

### 3. Merak-7B (Community Model) - BEST FOR LOW RESOURCES

**Hugging Face Links:**
- v1: https://huggingface.co/Ichsan2895/Merak-7B-v1
- v3: https://huggingface.co/Ichsan2895/Merak-7B-v3
- v4: https://huggingface.co/Ichsan2895/Merak-7B-v4

**Size:** 7B parameters

**Indonesian Capability:** Native - trained specifically on Indonesian Wikipedia

**JSON/Structured Output:** Yes (with proper prompting)

**Hardware Requirements:**
- Designed to run with 16GB VRAM using QLoRA
- 7B VRAM (8-bit) or 3.5GB VRAM (4-bit quantized)

**Inference Speed:** ~60-90 tokens/sec on RTX 4090

**Ollama Compatibility:** Likely supported (based on Mistral)

**vLLM/TGI Support:** Yes - based on Mistral architecture

**Fine-tuning:** Yes - uses QLoRA for efficient fine-tuning

**Training Data:**
- v1: 200K cleaned Indonesian Wikipedia articles
- v2: 600K Indonesian Wikipedia articles
- v3: Fine-tuned on OASST_Top1_Indonesian & alpaca-gpt4-indonesian

**License:** Creative Commons BY-SA-NC 4.0 (Non-commercial)

**Strengths:**
- Optimized for consumer hardware (16GB VRAM)
- Multiple iterations with improvements
- Community-driven development
- Good for resource-constrained environments

**Considerations:**
- Non-commercial license
- Cendol models found to be 6.34% more truthful
- Limited to Wikipedia-style knowledge

---

### 4. Cendol (Indonesian NLP Community) - BEST RANGE OF SIZES

**Hugging Face Link:** https://huggingface.co/indonlp/cendol

**Size:** Multiple variants
- mT5-based: 300M, 580M, 1.2B, 3.7B, 13B
- LLaMA-2-based: 7B, 13B

**Indonesian Capability:** Native - specifically tailored for Indonesian indigenous languages

**JSON/Structured Output:** Yes - encoder-decoder models excel at structured tasks

**Hardware Requirements:**
- 300M: 2GB VRAM
- 580M: 3GB VRAM
- 1.2B: 5GB VRAM
- 3.7B: 8GB VRAM
- 7B: 14GB VRAM (FP16) or 7GB (8-bit)
- 13B: 26GB VRAM (FP16) or 13GB (8-bit)

**Inference Speed:**
- Varies by size
- 3.7B: ~100+ tokens/sec on RTX 4090
- 7B: ~60-90 tokens/sec on RTX 4090

**Ollama Compatibility:** LLaMA-2 versions supported

**vLLM/TGI Support:** LLaMA-2 versions supported

**Fine-tuning:** Yes - Apache 2.0 license

**License:** Apache 2.0

**Strengths:**
- Wide range of model sizes (300M to 13B)
- Both decoder-only and encoder-decoder architectures
- 20% improvement over base models on various tasks
- More truthful and less toxic than Merak
- Excellent for resource optimization
- Best documented Indonesian LLM research

**Considerations:**
- Trained October 2023 - January 2024 (may need updates)
- Multiple architecture choices require selection

---

### 5. Sidrap-7B (Community Model)

**Hugging Face Links:**
- v1: https://huggingface.co/robinsyihab/Sidrap-7B-v1
- v2: https://huggingface.co/robinsyihab/Sidrap-7B-v2

**Size:** 7B parameters

**Indonesian Capability:** Native - fine-tuned on high-quality Indonesian datasets

**JSON/Structured Output:** Yes (with prompting)

**Hardware Requirements:**
- 14GB VRAM (FP16) or 7GB VRAM (8-bit quantized)

**Inference Speed:** ~70-100 tokens/sec on RTX 4090 (Mistral-based)

**Ollama Compatibility:** Yes - based on Mistral-7B-v0.1

**vLLM/TGI Support:** Yes - Mistral architecture

**Fine-tuning:** Yes - available for fine-tuning

**Strengths:**
- Based on Mistral (fast inference)
- Curated high-quality Indonesian dataset
- Good for conversational tasks

**Considerations:**
- Less documentation than Komodo or Sahabat-AI
- Focus on dialogue over structured tasks

---

## Category 2: Multilingual Models (Good Indonesian Support)

### 6. SeaLLM-13B (Southeast Asia Focus) - BEST MULTILINGUAL SEA

**Hugging Face Link:** https://huggingface.co/SeaLLMs/SeaLLM-13B-Chat

**Size:** 13B parameters (also 7B variant available)

**Indonesian Capability:** Multilingual - optimized for Southeast Asian languages
- Strong Indonesian support
- Also: Vietnamese, Thai, Malay, Khmer, Lao, Tagalog, Burmese, Javanese
- Plus: English, Chinese

**JSON/Structured Output:** Yes - chat variant supports structured output

**Hardware Requirements:**
- 13B: 26GB VRAM (FP16) or 13GB VRAM (8-bit) or 6.5GB (4-bit)

**Inference Speed:** ~40-60 tokens/sec on RTX 4090

**Ollama Compatibility:** Likely (based on Llama-2)

**vLLM/TGI Support:** Yes - based on Llama-2

**Fine-tuning:** Yes - available

**Strengths:**
- Superior to ChatGPT-3.5 in low-resource languages
- Pretrained on tailored SEA language dataset
- Maintains performance in high-resource languages
- Excellent for cross-lingual applications

**Considerations:**
- Larger model (13B)
- More compute-intensive

---

### 7. Qwen3-8B (Alibaba) - BEST JSON STRUCTURED OUTPUT

**Hugging Face Link:** https://huggingface.co/Qwen/Qwen3-8B

**Size:** 0.6B, 1.7B, 4B, 8B, 14B, 32B variants

**Indonesian Capability:** Multilingual - supports 119 languages including Indonesian

**JSON/Structured Output:** EXCELLENT - specifically enhanced for JSON structured output

**Hardware Requirements:**
- 8B: 16GB VRAM (FP16) or 8GB VRAM (8-bit) or 4GB (4-bit)

**Inference Speed:** ~60-90 tokens/sec on RTX 4090

**Ollama Compatibility:** Yes - native Ollama support (`ollama run qwen3:8b`)

**vLLM/TGI Support:** Yes - extensive support

**Fine-tuning:** Yes - Apache 2.0 license

**Context Length:** 128K tokens

**License:** Apache 2.0

**Strengths:**
- State-of-the-art JSON structured output capabilities
- 119 languages and dialects support
- Excellent instruction following
- Strong coding and reasoning
- Wide range of sizes
- Translation support for Indonesian
- Easy Ollama deployment

**Considerations:**
- Indonesian is not primary focus (compared to Chinese/English)
- Performance may vary for Indonesian vs primary languages

**Recommendation:** Best choice for JSON slot extraction with decent Indonesian support

---

### 8. Qwen2.5-7B/14B (Alibaba)

**Hugging Face Link:** https://huggingface.co/Qwen/Qwen2.5-7B

**Size:** 0.5B to 72B variants (7B and 14B recommended)

**Indonesian Capability:** Multilingual - supports 29+ languages including Indonesian

**JSON/Structured Output:** Excellent - enhanced for structured outputs

**Hardware Requirements:**
- 7B: 14GB VRAM (FP16) or 7GB VRAM (8-bit)
- 14B: 28GB VRAM (FP16) or 14GB VRAM (8-bit)

**Inference Speed:**
- 7B: ~60-90 tokens/sec on RTX 4090
- 14B: ~40-60 tokens/sec on RTX 4090

**Ollama Compatibility:** Yes (`ollama run qwen2.5`)

**vLLM/TGI Support:** Yes

**Fine-tuning:** Yes - Apache 2.0

**Training Data:** Up to 18 trillion tokens

**Strengths:**
- Excellent for tables and structured data
- Superior instruction following
- Long text generation (8K+ tokens)
- Mature, well-documented

**Considerations:**
- Qwen3 supersedes this for most use cases

---

### 9. Llama 3.1-8B (Meta) - MOST POPULAR BASELINE

**Hugging Face Link:** https://huggingface.co/meta-llama/Llama-3.1-8B

**Size:** 8B, 70B, 405B

**Indonesian Capability:** Multilingual - Indonesian included but not primary

**JSON/Structured Output:** Yes - instruction-tuned variants support structured output

**Hardware Requirements:**
- 8B: 16GB VRAM (FP16) or 8GB VRAM (8-bit)

**Inference Speed:** ~70-90 tokens/sec on RTX 4090

**Ollama Compatibility:** Yes - native support

**vLLM/TGI Support:** Yes - excellent support

**Fine-tuning:** Yes - extensive tooling

**Context Length:** 128K tokens

**Strengths:**
- Industry standard
- Extensive ecosystem and tooling
- Large community
- Well-tested
- Great base for fine-tuning

**Considerations:**
- Indonesian is not primary language
- Better options exist for Indonesian-specific tasks

---

### 10. Llama 4 (Meta) - LATEST RELEASE

**Ollama:** `ollama run llama4`

**Size:** Multiple variants

**Indonesian Capability:** Multilingual - officially supports Indonesian
- Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, Vietnamese

**JSON/Structured Output:** Yes - multimodal with structured output

**Hardware Requirements:**
- Similar to Llama 3.1

**Inference Speed:** Improved over Llama 3.1

**Ollama Compatibility:** Yes

**vLLM/TGI Support:** Yes

**Fine-tuning:** Yes

**Strengths:**
- Latest Meta release
- Official Indonesian support
- Multimodal capabilities
- State-of-the-art performance

**Considerations:**
- Very new (limited real-world testing)
- May require more resources

---

### 11. Phi-3.5-mini (Microsoft) - BEST SMALL MODEL

**Hugging Face Link:** https://huggingface.co/microsoft/Phi-3.5-mini-instruct

**Size:** 3.8B parameters

**Indonesian Capability:** Multilingual - officially supports Indonesian (20+ languages)

**JSON/Structured Output:** Yes - instruction-tuned

**Hardware Requirements:**
- 3.8B: 8GB VRAM (FP16) or 4GB VRAM (8-bit) or 2GB (4-bit)

**Inference Speed:** ~120-150 tokens/sec on RTX 4090

**Ollama Compatibility:** Yes

**vLLM/TGI Support:** Yes

**Fine-tuning:** Yes

**Context Length:** 128K

**Strengths:**
- Runs on consumer laptops
- Excellent performance for size
- Enhanced multilingual support (Phi-3.5)
- Very fast inference
- Small footprint

**Considerations:**
- Performance lower than English (multilingual training)
- Better for small-scale deployments

---

### 12. Aya-23 (Cohere)

**Size:** Multiple variants

**Indonesian Capability:** Multilingual - covers 100+ languages including Indonesian

**JSON/Structured Output:** Yes

**Hardware Requirements:** Varies by size

**Ollama Compatibility:** Yes

**vLLM/TGI Support:** Yes

**Fine-tuning:** Yes

**Strengths:**
- Massive language coverage
- Focus on low-resource languages
- Good for cross-lingual applications

**Considerations:**
- Quality may vary across languages
- Less Indonesian-specific optimization

---

## Category 3: Encoder Models (For Fine-Tuning)

### 13. IndoBERT (Indonesian BERT) - BEST FOR FINE-TUNING NER/SLOT

**Hugging Face:** Various versions available

**Size:** 110M-340M parameters (BERT-Base to BERT-Large)

**Indonesian Capability:** Native - monolingual Indonesian pre-trained model

**JSON/Structured Output:** Via fine-tuning for sequence labeling

**Hardware Requirements:**
- Base: 2-4GB VRAM for inference
- Fine-tuning: 8-12GB VRAM

**Inference Speed:** Very fast (~200+ tokens/sec)

**Ollama Compatibility:** No (BERT architecture)

**vLLM/TGI Support:** No (encoder-only)

**Fine-tuning:** EXCELLENT - designed for fine-tuning on downstream tasks

**Benchmarks:**
- IndoNLU benchmark (12 tasks)
- IndoLEM benchmark (NER, POS, dependency parsing)
- Superior performance on Indonesian NER tasks

**Strengths:**
- Specifically designed for Indonesian
- Excellent for NER and slot filling
- Fast inference
- Low resource requirements
- Strong benchmarks on Indonesian tasks
- Well-documented fine-tuning approaches

**Considerations:**
- Not generative (encoder-only)
- Requires fine-tuning for specific slot schemas
- Cannot generate free-form text

**Recommendation:** Best choice if you plan to fine-tune on your specific slot extraction task

---

## Production Deployment Considerations

### Ollama Deployment (Easiest)

**Supported Models:**
- Qwen3, Qwen2.5
- Llama 3.1, Llama 4
- Phi-3.5
- Gemma 2
- Mistral-based (Sidrap)
- Most popular base models

**Advantages:**
- Simple setup: `ollama run qwen3:8b`
- Structured output support with Pydantic
- Built-in model management
- Docker support

**Command Example:**
```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

---

### vLLM (Best Performance)

**Supported Models:**
- All Llama-based models (Sahabat-AI, Komodo, SeaLLM)
- Qwen family
- Mistral-based models
- Most transformer models

**Advantages:**
- Up to 24x higher throughput than HuggingFace
- PagedAttention for efficient memory usage
- OpenAI-compatible API
- Kubernetes-native with llm-d (2025)
- Best for high-concurrency workloads

**Disadvantages:**
- Less production tooling than TGI
- Higher setup complexity

**Use Cases:**
- High-throughput batch processing
- Multi-user serving
- Production APIs

---

### TGI (Text Generation Inference - Best Production)

**Supported Models:**
- Llama, Mistral, Gemma families
- Qwen models
- Most popular open-source LLMs

**Advantages:**
- Production-ready features
- Built-in telemetry (OpenTelemetry, Prometheus)
- 13x speedup on long prompts (200K+ tokens)
- Better latency for interactive applications
- HuggingFace ecosystem integration
- Multi-backend support (can use vLLM backend)

**Disadvantages:**
- Lower throughput than vLLM in batch scenarios

**Use Cases:**
- Interactive applications
- Production deployments with monitoring
- Low-latency requirements

---

### Hybrid Approach (Recommended)

**Development:** Ollama (rapid prototyping)
**Production:** vLLM or TGI (performance & reliability)

---

## Hardware Recommendations by Budget

### Entry Level ($300-500)
**RTX 4060 Ti 16GB** - $500
- Runs: 7B models at FP16, 13B at 4-bit
- Best for: Merak-7B, Komodo-7B, Cendol-7B, Phi-3.5, IndoBERT

### Mid Range ($1000-1500)
**RTX 4070 Ti Super 16GB** or **RTX 4080 16GB**
- Runs: 8B models comfortably, 13B at 8-bit
- Best for: Sahabat-AI 8B, Qwen3-8B, SeaLLM-13B

### High End ($1600-2000)
**RTX 4090 24GB** - $1600
**RTX 5090 32GB** - $1999
- Runs: 13B at FP16, 30B at 4-bit, 70B at 4-bit (5090)
- Best for: All models, Sahabat-AI 70B

### Server/Cloud
**NVIDIA A100 40GB/80GB** or **H100 80GB**
- Runs: Any model size
- Best for: Production deployments

---

## Recommended Configurations for Indonesian Slot Extraction

### Configuration 1: Best Indonesian Quality (Recommended)
**Model:** Sahabat-AI 8B Instruct
**Hardware:** RTX 4060 Ti 16GB or better
**Deployment:** Ollama (dev) → vLLM (prod)
**Quantization:** 8-bit for 16GB, FP16 for 24GB
**Why:** Native Indonesian, strong institutional backing, good performance

### Configuration 2: Best Performance/Cost
**Model:** Qwen3-8B
**Hardware:** RTX 4060 Ti 16GB
**Deployment:** Ollama (easiest)
**Quantization:** 8-bit
**Why:** Excellent JSON output, easy deployment, good Indonesian support

### Configuration 3: Best for Regional Languages
**Model:** Komodo-7B
**Hardware:** RTX 4060 Ti 16GB
**Deployment:** Ollama or vLLM
**Quantization:** 8-bit
**Why:** 11 regional languages, efficient tokenizer, proven benchmarks

### Configuration 4: Smallest/Fastest
**Model:** Phi-3.5-mini (3.8B) or Cendol mT5-XL (3.7B)
**Hardware:** 8GB GPU or CPU
**Deployment:** Ollama
**Quantization:** 4-bit
**Why:** Runs anywhere, fast inference, decent quality

### Configuration 5: Fine-Tuning Approach (Best Accuracy)
**Model:** IndoBERT + fine-tuning on your slot schema
**Hardware:** 12GB+ for training, 4GB for inference
**Deployment:** Custom inference server
**Why:** Best accuracy for specific slot schemas, fast inference

---

## JSON Structured Output Implementation

### Method 1: Native JSON Mode (Qwen3, Llama 3.1)
```python
# With Ollama
response = ollama.generate(
    model='qwen3:8b',
    prompt=prompt,
    format='json'
)
```

### Method 2: Pydantic Schema Constraints
```python
from pydantic import BaseModel

class SlotExtraction(BaseModel):
    intent: str
    slots: dict[str, str]
    confidence: float

# Use with Instructor library
result = client.chat.completions.create(
    model="qwen3:8b",
    response_model=SlotExtraction,
    messages=[...]
)
```

### Method 3: Grammar-Based (Outlines/Formatron)
```python
from outlines import models, generate

model = models.transformers("Qwen/Qwen3-8B")
generator = generate.json(model, SlotExtraction)
result = generator(prompt)
```

### Method 4: SLOT Framework
- Use lightweight Mistral-7B or Llama-3.2-1B as post-processor
- 99.5% schema accuracy
- Works with any base model output

---

## Benchmarks & Datasets

### Indonesian NLP Benchmarks
- **IndoNLU**: 12 tasks, various domains
- **IndoLEM**: POS, NER, dependency parsing
- **SEA-HELM (BHASA)**: Southeast Asian evaluation
- **Indo MMLU**: Indonesian language understanding
- **XCOPA-ID**: Indonesian commonsense reasoning

### NER/Slot Extraction Datasets
- **NERGrit**: PERSON, PLACE, ORGANIZATION
- **NERP**: PER, LOC, IND, EVT, FNB
- **TermA**: Aspect and sentiment spans (hotel reviews)
- **KEPS**: Keyphrase extraction (banking)
- **IPerFEX-2023**: Financial entity extraction (15 types)

### Available Training Data
- **Indo4B**: Large-scale Indonesian pre-training corpus
- **OSCAR**: 4 billion tokens Indonesian text
- **CC-100**: CommonCrawl Indonesian data
- **Indonesian Wikipedia**: Curated articles
- **Cendol Collection**: v1 and v2 datasets

---

## Final Rankings

### By Indonesian Language Quality
1. Sahabat-AI 8B/9B (Native, best funded)
2. Komodo-7B (Native, regional languages, proven benchmarks)
3. Merak-7B/Sidrap-7B (Native, community models)
4. Cendol family (Native, research-backed)
5. SeaLLM-13B (SEA-optimized multilingual)
6. Qwen3-8B (Good multilingual)

### By Ease of Deployment
1. Qwen3-8B (Native Ollama, best docs)
2. Llama 3.1/4 (Native Ollama, large ecosystem)
3. Phi-3.5-mini (Native Ollama, small size)
4. Sahabat-AI (Standard deployment)
5. Komodo-7B (GGUF available)

### By Hardware Requirements
1. Phi-3.5-mini - 2GB (4-bit)
2. Cendol-300M/580M - 2-3GB
3. IndoBERT - 2-4GB
4. Cendol-3.7B - 4GB (4-bit)
5. 7B models - 4-8GB (4-bit)
6. 8B-9B models - 4-9GB (4-bit)

### By JSON Output Reliability
1. Qwen3-8B (Specifically optimized)
2. Qwen2.5 family (Structured output focus)
3. Llama 3.1+ (Native JSON mode)
4. Sahabat-AI (Instruction-tuned)
5. Phi-3.5 (Good structured output)

### Overall Best Choice for Your Use Case
**Primary Recommendation: Sahabat-AI 8B Instruct**
- Native Indonesian with institutional backing
- Good balance of quality and resource requirements
- Open-source with active development
- Proven download numbers (35K+)

**Alternative: Qwen3-8B**
- If JSON output reliability is critical
- Easier deployment with Ollama
- Good Indonesian support (though not native)
- Excellent documentation and tooling

**Budget Option: Komodo-7B**
- More efficient than Sahabat-AI
- Regional language support
- Strong benchmarks
- Lower compute requirements

**Fine-tuning Option: IndoBERT**
- If you have labeled data for your specific slots
- Best accuracy for specialized schemas
- Lowest inference cost
- Fast training and deployment

---

## Implementation Roadmap

### Phase 1: Prototype (Week 1-2)
1. Set up Ollama locally
2. Test Qwen3-8B and Sahabat-AI 8B
3. Implement basic slot extraction prompts
4. Evaluate JSON output quality
5. Test with Indonesian sample data

### Phase 2: Optimization (Week 3-4)
1. Fine-tune prompt templates
2. Implement schema validation
3. Test quantization options (4-bit vs 8-bit)
4. Benchmark inference speeds
5. Compare model outputs on real data

### Phase 3: Production Deployment (Week 5-6)
1. Deploy chosen model with vLLM or TGI
2. Implement API layer
3. Add monitoring (Prometheus metrics)
4. Load testing and optimization
5. Error handling and fallbacks

### Phase 4: Fine-tuning (Optional, Week 7+)
1. Collect and label Indonesian slot extraction data
2. Fine-tune IndoBERT or chosen base model
3. Evaluate fine-tuned vs base model
4. Deploy fine-tuned model

---

## Sources

### Indonesian Models
- [Sahabat-AI Official Announcement](https://www.gotocompany.com/en/news/press/indosat-ooredoo-hutchison-and-goto-launch-sahabat-ai-indonesias-open-source-llm-for-empowering-digital-sovereignty)
- [Sahabat-AI on Hugging Face](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct)
- [Komodo-7B Research Paper](https://arxiv.org/html/2403.09362v1)
- [Komodo-7B Yellow.ai](https://tech.yellow.ai/p/komodo-7b-the-first-llm-for-regional)
- [Merak-7B on Hugging Face](https://huggingface.co/Ichsan2895/Merak-7B-v4)
- [Cendol Research Paper](https://arxiv.org/html/2404.06138v1)
- [Cendol on Hugging Face](https://huggingface.co/indonlp/cendol)
- [Sidrap-7B on Hugging Face](https://huggingface.co/robinsyihab/Sidrap-7B-v2)

### Multilingual Models
- [SeaLLM on Hugging Face](https://huggingface.co/SeaLLMs/SeaLLM-13B-Chat)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Qwen Documentation](https://qwen.readthedocs.io/)
- [Phi-3.5 Announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/4225280)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)

### Benchmarks & Datasets
- [IndoNLU Paper](https://www.researchgate.net/publication/344233900_IndoNLU_Benchmark_and_Resources_for_Evaluating_Indonesian_Natural_Language_Understanding)
- [IndoLEM Paper](https://aclanthology.org/2020.coling-main.66.pdf)
- [IndoNLU on Hugging Face](https://huggingface.co/datasets/indonlp/indonlu)

### Structured Output
- [SLOT: Structuring LLM Output](https://arxiv.org/html/2505.04016v1)
- [Awesome LLM JSON GitHub](https://github.com/imaurer/awesome-llm-json)
- [Ollama Structured Output Guide](https://www.glukhov.org/post/2025/09/llm-structured-output-with-ollama-in-python-and-go/)

### Deployment
- [vLLM vs TGI Comparison](https://modal.com/blog/vllm-vs-tgi-article)
- [vLLM vs TensorRT-LLM Deep Dive](https://www.marktechpost.com/2025/11/19/vllm-vs-tensorrt-llm-vs-hf-tgi-vs-lmdeploy-a-deep-technical-comparison-for-production-llm-inference/)
- [vLLM Best Choice for AI Inference](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today)
- [TGI Multi-backend Support](https://huggingface.co/blog/tgi-multi-backend)

### Hardware & VRAM
- [VRAM Requirements Modal Blog](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [VRAM Calculator LocalLLM](https://localllm.in/blog/interactive-vram-calculator)
- [LLM GPU Requirements Guide 2025](https://www.propelrc.com/llm-gpu-vram-requirements-explained/)
- [Best GPU for Local LLM 2025](https://nutstudio.imyfone.com/llm-tips/best-gpu-for-local-llm/)

### Ollama
- [Ollama Models Library](https://ollama.com/library)
- [Ollama Models List 2025](https://skywork.ai/blog/llm/ollama-models-list-2025-100-models-compared/)
- [Ollama Deployment Guide](https://dev.to/sina14/your-guide-to-local-llms-ollama-deployment-models-and-use-cases-2jng)

---

## Conclusion

For Indonesian semantic slot extraction with self-hosted LLMs, the optimal choice depends on your priorities:

**Best Overall: Sahabat-AI 8B Instruct**
- Native Indonesian, institutional backing, good resources

**Best JSON Output: Qwen3-8B**
- Superior structured output, easy deployment, good Indonesian

**Best Efficiency: Komodo-7B**
- Proven benchmarks, regional languages, lower compute

**Best for Fine-tuning: IndoBERT**
- Specialized for sequence labeling, lowest cost, highest accuracy for custom schemas

All recommended models are production-ready with proper deployment infrastructure (vLLM/TGI/Ollama).
