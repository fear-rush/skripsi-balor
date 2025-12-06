# Research Analysis: Techniques to Address High Accuracy on Synthetic Data

## Executive Summary

Your Indonesian e-commerce intent classifier is achieving 98% accuracy on synthetic V4 data despite efforts to make it harder (shared keywords, hard negatives, generic patterns). This is a classic symptom of **models learning template patterns rather than semantic understanding**. While Epoch 1 showed promise (44% vs 84% in V3), rapid convergence to 98% by Epoch 2-3 indicates the model is memorizing distributional artifacts rather than learning robust intent representations.

**Key Insight**: You need techniques that force the model to focus on **harder examples**, **prevent overconfidence**, and **ensure semantic diversity** beyond template variations.

---

## Detailed Techniques Analysis

### 1. Label Smoothing and Soft Labels

#### How It Works
Label smoothing converts hard one-hot labels (e.g., [0, 1, 0, 0]) into soft distributions (e.g., [0.025, 0.925, 0.025, 0.025]). This is controlled by a smoothing parameter ε (typically 0.1), where the target becomes:

```
y_smooth = (1 - ε) * y_hard + ε / num_classes
```

The smoothing prevents the model from becoming overconfident and adds implicit regularization.

#### Relevance to Your Problem
**HIGH RELEVANCE** - Label smoothing directly addresses overconfidence on template-based data:
- Forces model to maintain uncertainty even on "easy" template matches
- Prevents the model from learning to assign 100% confidence to patterns
- Improves calibration and generalization to real-world variations
- Learning label smoothing (LLS) adapts the smoothing rate per sample, which could help with your hard negatives

#### Implementation Complexity
**LOW** - Can be implemented in 2-5 lines of code:

```python
# Simple uniform label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Convert targets to one-hot then smooth
        targets_one_hot = F.one_hot(targets, n_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes

        loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
        return loss
```

For Hugging Face Trainer:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    label_smoothing_factor=0.1,  # Built-in support
    # ... other args
)
```

#### Expected Impact
- **Accuracy**: Should drop from 98% to **88-93%** on synthetic test set
- **Real-world performance**: Likely to improve generalization by 3-7%
- **Calibration**: Significantly better confidence calibration

**Recommendation**: Start with ε=0.1, experiment with 0.05-0.2. Consider adaptive label smoothing (LLS) that learns different smoothing rates for hard vs easy examples.

**Sources**:
- [Learning label smoothing for text classification - PeerJ](https://peerj.com/articles/cs-2005/)
- [Label Smoothing for Enhanced Text Sentiment Classification](https://arxiv.org/html/2312.06522v1)
- [Understanding Why Label Smoothing Degrades Selective Classification](https://arxiv.org/html/2403.14715v1)

---

### 2. Contrastive Learning Approaches

#### How It Works
Contrastive learning trains models to distinguish similar vs dissimilar examples by learning an embedding space where:
- Similar intents are pulled together
- Different intents are pushed apart
- Hard negatives (similar but different intents) get special attention

The loss function (e.g., NT-Xent, SupCon) encourages:
```
L = -log(exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, negative_i) / τ))
```

Where τ is a temperature parameter controlling separation strength.

#### Relevance to Your Problem
**VERY HIGH RELEVANCE** - This directly addresses your hard negatives problem:
- **Sample-level contrastive learning**: Compares different augmentations of the same query
- **Class-level contrastive learning**: Prevents overfitting to known class patterns
- **MICL (Mutual Information + Contrastive Learning)**: Specifically designed for few-shot intent detection
- Forces model to learn **discriminative intent keywords** rather than template patterns

#### Implementation Complexity
**MEDIUM-HIGH**:

```python
# Supervised Contrastive Loss for Intent Classification
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # Mask out self-similarity
        mask.fill_diagonal_(0)

        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (1 - torch.eye(len(features), device=features.device))

        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = -mean_log_prob.mean()
        return loss

# Usage in training
class ContrastiveIntentClassifier(nn.Module):
    def __init__(self, base_model, num_intents, projection_dim=128):
        super().__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.classifier = nn.Linear(base_model.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get base representations
        outputs = self.encoder(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token

        # For contrastive learning
        projections = self.projection_head(pooled)

        # For classification
        logits = self.classifier(pooled)

        if labels is not None:
            # Combined loss
            ce_loss = F.cross_entropy(logits, labels)
            contrastive_loss = SupervisedContrastiveLoss()(projections, labels)
            loss = ce_loss + 0.5 * contrastive_loss  # Weight can be tuned
            return loss, logits

        return logits
```

For simpler implementation, you can use **SimCSE-style** approach:
- Apply dropout twice to get two views of the same input
- Minimize distance between views from same sample
- Maximize distance between different samples

#### Expected Impact
- **Accuracy**: Should stabilize at **85-91%** on synthetic data
- **Hard negative performance**: Significant improvement (15-25% better)
- **Representation quality**: Much better semantic clustering
- **Training time**: 1.5-2x longer due to additional loss computation

**Key Benefit**: Forces model to learn "what makes intents different" rather than just "what templates match".

**Sources**:
- [Few-shot intent detection with mutual information and contrastive learning](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011128)
- [Contrastive Learning in NLP](https://medium.com/data-scientists-diary/contrastive-learning-in-nlp-dc0e1a5bb23c)
- [The Beginner's Guide to Contrastive Learning](https://www.v7labs.com/blog/contrastive-learning-guide)

---

### 3. Data Augmentation Beyond Templates

#### How It Works
Moving beyond simple template filling to semantic variations:

**a) Paraphrasing with Pre-trained Models** (Indonesian-specific):
- **mBART50** (fine-tuned): Best for Indonesian, balances semantic preservation and lexical diversity
- **IndoBART-v2**: Indonesian-specific BART model
- **IndoGPT2**: Indonesian GPT-2 variant

**b) Back-translation**:
- Indonesian → English → Indonesian
- Generates natural variations but struggles with informal text

**c) Contextual Word Substitution**:
- Replace words with contextually appropriate synonyms using BERT-like models
- More subtle than random synonym replacement

#### Relevance to Your Problem
**MEDIUM-HIGH RELEVANCE** - Can help break template patterns:
- Paraphrasing creates **genuinely different** surface forms for same intent
- Unlike templates, paraphrasing can change sentence structure, word order, formality
- Research shows fine-tuned mBART50 significantly improves F1 for minority labels in Indonesian classification

**However**: Won't solve the fundamental problem if model still finds distributional shortcuts. Best combined with other techniques.

#### Implementation Complexity
**MEDIUM**:

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load fine-tuned Indonesian paraphrase model
model = MBartForConditionalGeneration.from_pretrained("your-finetuned-mbart50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

def paraphrase_text(text, num_variations=3):
    tokenizer.src_lang = "id_ID"
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate multiple paraphrases with sampling
    outputs = model.generate(
        **inputs,
        num_return_sequences=num_variations,
        num_beams=5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        forced_bos_token_id=tokenizer.lang_code_to_id["id_ID"]
    )

    paraphrases = [tokenizer.decode(output, skip_special_tokens=True)
                   for output in outputs]
    return paraphrases

# Example: Augment training data
augmented_data = []
for query, intent in original_data:
    # Keep original
    augmented_data.append((query, intent))

    # Add paraphrases
    for para in paraphrase_text(query, num_variations=2):
        if para != query:  # Avoid duplicates
            augmented_data.append((para, intent))
```

**Alternative: EDA (Easy Data Augmentation) + Back-translation**:
```python
# Simpler approach without heavy models
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Contextual word embeddings augmentation
aug_bert = naw.ContextualWordEmbsAug(
    model_path='indobenchmark/indobert-base-p1',
    action="substitute",
    aug_p=0.15  # Replace 15% of words
)

# Back-translation
aug_bt = nas.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-id-en',
    to_model_name='Helsinki-NLP/opus-mt-en-id'
)

augmented_query = aug_bert.augment(original_query)
backtranslated = aug_bt.augment(original_query)
```

#### Expected Impact
- **Data diversity**: 2-3x more diverse surface forms
- **Accuracy on synthetic**: May paradoxically increase (93-98%) if augmented data maintains patterns
- **Real-world performance**: +5-10% improvement on actual e-commerce queries
- **Training time**: 2-3x longer with augmented dataset

**Recommendation**: Use as a complement, not primary solution. Best combined with regularization techniques.

**Sources**:
- [Study of Text Augmentation with Paraphrasing for Indonesian](https://join.if.uinsgd.ac.id/index.php/join/article/view/1472)
- [Data Augmentation by Backtranslation](https://github.com/vietai/dab)
- [IndoNLG: Indonesian NLG Benchmark](https://github.com/IndoNLP/indonlg)

---

### 4. Curriculum Learning Strategies

#### How It Works
Train the model progressively from easier to harder examples:

**a) Static Curriculum**: Pre-define difficulty levels (e.g., Level 1: distinct keywords → Level 5: maximum overlap)

**b) Dynamic Curriculum (SPDCL)**: Adaptively reorder and resample data based on:
- Model confidence on each sample
- Linguistic complexity
- Loss magnitude

**c) Hard Negative Curriculum**: Gradually introduce harder negatives
- Start: Train on intents with distinct keywords
- Middle: Introduce intents with some shared terms
- End: Full hard negative training

#### Relevance to Your Problem
**VERY HIGH RELEVANCE** - Directly addresses your rapid convergence issue:
- Your Epoch 1 results (44%) suggest the model struggles initially with V4's hard negatives
- Jumping to 92% by Epoch 2 suggests it finds shortcuts rather than learning properly
- Curriculum learning can prevent this by ensuring solid learning before hard examples

**Multi-Granularity Hard-Negative Synthesis**:
- Generates negatives at multiple difficulty levels
- Implements coarse-to-fine progression
- Has shown 24% improvement on difficult benchmarks

#### Implementation Complexity
**MEDIUM**:

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

class CurriculumSampler(Sampler):
    def __init__(self, dataset, difficulty_scores, epoch, total_epochs):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores  # 0 (easy) to 1 (hard)
        self.epoch = epoch
        self.total_epochs = total_epochs

    def __iter__(self):
        # Calculate difficulty threshold for this epoch
        # Start with easiest 50%, gradually include harder examples
        threshold = 0.5 + (self.epoch / self.total_epochs) * 0.5

        # Get indices of samples within difficulty range
        valid_indices = np.where(self.difficulty_scores <= threshold)[0]

        # Shuffle and return
        np.random.shuffle(valid_indices)
        return iter(valid_indices)

    def __len__(self):
        threshold = 0.5 + (self.epoch / self.total_epochs) * 0.5
        return np.sum(self.difficulty_scores <= threshold)

# Define difficulty based on keyword overlap with other intents
def compute_difficulty_scores(dataset, intent_keywords):
    """
    Higher difficulty = more keyword overlap with other intents
    """
    difficulty_scores = []

    for query, intent in dataset:
        query_words = set(query.lower().split())

        # Count overlap with other intents' keywords
        max_overlap = 0
        for other_intent, keywords in intent_keywords.items():
            if other_intent != intent:
                overlap = len(query_words & set(keywords))
                max_overlap = max(max_overlap, overlap)

        # Normalize to 0-1 range
        difficulty = min(max_overlap / 5.0, 1.0)  # Cap at 5 overlapping words
        difficulty_scores.append(difficulty)

    return np.array(difficulty_scores)

# Training loop with curriculum
for epoch in range(num_epochs):
    difficulty_scores = compute_difficulty_scores(train_dataset, intent_keywords)
    sampler = CurriculumSampler(train_dataset, difficulty_scores, epoch, num_epochs)
    dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    # Train as usual
    for batch in dataloader:
        # ... training code
```

**Dynamic approach (self-paced)**:
```python
class DynamicCurriculumTrainer:
    def __init__(self, model, initial_pace=0.5):
        self.model = model
        self.pace = initial_pace  # What fraction of hardest examples to include
        self.sample_weights = None

    def update_curriculum(self, losses):
        # Compute sample-wise difficulty from recent losses
        difficulty = losses / losses.max()

        # Adaptive pace: if model is doing well, increase pace
        avg_loss = losses.mean()
        if avg_loss < self.prev_loss * 0.9:  # 10% improvement
            self.pace = min(1.0, self.pace + 0.1)

        # Weight samples: down-weight hardest (1-pace)% of examples
        threshold = np.percentile(difficulty, (1 - self.pace) * 100)
        weights = torch.where(difficulty <= threshold,
                             torch.ones_like(difficulty),
                             torch.ones_like(difficulty) * 0.1)  # Reduce weight of hard examples

        self.sample_weights = weights
        self.prev_loss = avg_loss
```

#### Expected Impact
- **Accuracy progression**: Smoother learning curve (Epoch 1: 55%, Epoch 2: 68%, Epoch 3: 78%, Final: 88%)
- **Final accuracy**: **85-89%** with better understanding
- **Stability**: More stable training, less prone to finding shortcuts
- **Training time**: ~same (just changes sample order)

**Key Benefit**: Prevents the model from "giving up" on hard examples early by finding template shortcuts.

**Sources**:
- [Multi-Granularity Hard-Negative Synthesis](https://arxiv.org/html/2509.00842)
- [Curriculum DPO on Synthetic Negatives](https://arxiv.org/abs/2505.17558)
- [Dynamic Curriculum Learning for Imbalanced Text Classification](https://arxiv.org/abs/2210.14724)

---

### 5. Regularization Techniques for Transformers

#### How It Works
Prevent the model from memorizing training patterns:

**a) Standard Techniques**:
- **Dropout** (0.1-0.3): Randomly zero out activations during training
- **Weight decay** (0.01): L2 penalty on weights
- **Layer normalization**: Stabilizes training, mild regularization effect

**b) Transformer-Specific**:
- **AttentionDrop**: Drops attention connections rather than neurons
  - Hard Attention Masking: Zeros top-k attention logits
  - Blurred Attention Smoothing: Gaussian smoothing over attention
  - Consistency Regularization: KL divergence between different attention masks

**c) R-Drop (Regularized Dropout)**:
- Apply dropout twice in same forward pass
- Minimize KL divergence between the two outputs
- Forces consistent predictions despite stochasticity

#### Relevance to Your Problem
**MEDIUM-HIGH RELEVANCE**:
- Your high accuracy suggests memorization of template patterns
- DistilBERT already has dropout, but increasing it may help
- R-Drop specifically prevents relying on any single feature path
- AttentionDrop prevents memorizing specific keyword attention patterns

#### Implementation Complexity
**LOW-MEDIUM**:

**Simple: Increase existing regularization**:
```python
from transformers import DistilBertForSequenceClassification, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained(
    'cahya/distilbert-base-indonesian',
    num_labels=num_intents,
    hidden_dropout_prob=0.2,  # Default is 0.1, increase to 0.2-0.3
    attention_probs_dropout_prob=0.2
)

training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization
    # ... other args
)
```

**Medium: Implement R-Drop**:
```python
class RDropTrainer(Trainer):
    def __init__(self, *args, rdrop_alpha=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # First forward pass
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        # Second forward pass (dropout will differ)
        outputs2 = model(**inputs)
        logits2 = outputs2.logits

        # Standard cross-entropy loss
        ce_loss = (outputs1.loss + outputs2.loss) / 2

        # KL divergence between two outputs (R-Drop)
        kl_loss = self.compute_kl_loss(logits1, logits2)

        # Combined loss
        loss = ce_loss + self.rdrop_alpha * kl_loss

        return (loss, outputs1) if return_outputs else loss

    def compute_kl_loss(self, logits1, logits2):
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)

        kl_loss = F.kl_div(p, q, reduction='none', log_target=True)
        kl_loss = kl_loss.sum(-1).mean()

        # Symmetric KL
        kl_loss2 = F.kl_div(q, p, reduction='none', log_target=True)
        kl_loss2 = kl_loss2.sum(-1).mean()

        return (kl_loss + kl_loss2) / 2
```

#### Expected Impact
- **Accuracy**: Drop to **89-93%** with increased dropout
- **Generalization**: +3-5% on real queries
- **Training stability**: May require more epochs to converge
- **R-Drop**: Particularly effective, can achieve **87-91%** with good generalization

**Recommendation**: Start with increased dropout (0.2), add R-Drop if needed. Use α=5.0 for R-Drop weight.

**Sources**:
- [AttentionDrop: Novel Regularization for Transformers](https://arxiv.org/html/2504.12088)
- [Advanced Regularization Protocols for Transformers](https://medium.com/@hassanbinabid/the-art-and-science-of-hyperparameter-optimization-in-llm-fine-tuning-f95bc6e9a80b)
- [Self-knowledge distillation via dropout](https://www.sciencedirect.com/science/article/abs/pii/S1077314223001005)

---

### 6. Alternative Model Architectures

#### How It Works
Use simpler models that have less capacity to memorize:

**a) CNN + BiLSTM Fusion**:
- Lightweight architecture (100x fewer parameters than BERT)
- Learns n-gram patterns + sequential context
- Can be compressed further through pruning

**b) Hierarchical Self-Attention (HiSAN)**:
- Self-attention like BERT but much simpler
- ~100x fewer parameters
- Less prone to memorization

**c) Lighter BERT variants**:
- **ALBERT**: 18M parameters vs DistilBERT's 67M (parameter sharing)
- **TinyBERT**: 14M parameters (knowledge distillation from BERT)

#### Relevance to Your Problem
**LOW-MEDIUM RELEVANCE**:
- DistilBERT (67M params) is already relatively lightweight
- Simpler models may help prevent memorization BUT will also hurt legitimate learning
- CNN models are "designed to memorize n-grams" which could worsen your problem
- Main benefit: Faster training for experimentation

**Counter-intuitive finding**: Research shows BERT often can't beat CNN/BiLSTM on some clinical text tasks, suggesting simpler models don't always guarantee better generalization.

#### Implementation Complexity
**LOW-MEDIUM**:

```python
# Simple CNN + BiLSTM baseline
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100,
                 filter_sizes=[3,4,5], hidden_dim=128, num_classes=10):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # CNN layers for n-gram features
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in filter_sizes
        ])

        # BiLSTM for sequential context
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Combine CNN and LSTM features
        total_features = len(filter_sizes) * num_filters + 2 * hidden_dim
        self.fc = nn.Linear(total_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # CNN features
        x_cnn = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        cnn_features = []
        for conv in self.convs:
            conv_out = F.relu(conv(x_cnn))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            cnn_features.append(pooled.squeeze(2))
        cnn_out = torch.cat(cnn_features, dim=1)

        # LSTM features
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state

        # Combine and classify
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)

        return logits

# Model size: ~1-2M parameters vs DistilBERT's 67M
```

#### Expected Impact
- **Accuracy**: Likely **82-88%** (lower than current but maybe better real-world)
- **Training speed**: 5-10x faster
- **Inference speed**: 10-20x faster
- **Generalization**: Unclear - simpler doesn't always mean better generalization

**Recommendation**: NOT your primary solution. Use only if:
1. You need fast iteration for hyperparameter tuning
2. You want a lightweight production model after solving the data problem

**Sources**:
- [Alternative non-BERT choices for low-resource languages](https://aclanthology.org/2022.deeplo-1.20/)
- [Limitations of Transformers on Clinical Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8387496/)

---

### 7. Advanced Evaluation Methods

#### How It Works
Measure generalization beyond simple held-out test accuracy:

**a) Few-Shot Evaluation**:
- Train on N-1 examples per intent
- Test on held-out examples
- Measures if model learns from minimal data or needs many examples to memorize

**b) Zero-Shot Cross-Validation**:
- Hold out entire intents during training
- Test if model can generalize to completely unseen intents using descriptions
- Gold standard for measuring true semantic understanding

**c) Adversarial Test Sets**:
- Manually curate queries that share maximum keywords with wrong intents
- Test if model relies on keywords or understands context

**d) Calibration Metrics**:
- **Expected Calibration Error (ECE)**: Are 80% confidence predictions actually correct 80% of the time?
- **Brier Score**: Measures both accuracy and calibration
- **Reliability diagrams**: Visualize confidence vs accuracy

#### Relevance to Your Problem
**VERY HIGH RELEVANCE**:
- 98% accuracy on synthetic test set tells you nothing about real performance
- These methods reveal whether high accuracy is due to:
  - True understanding (few-shot works well)
  - Pattern matching (few-shot fails)
  - Overconfidence (poor calibration)

**Critical insight**: Your model might be 98% accurate but 95% confident on wrong predictions!

#### Implementation Complexity
**LOW-MEDIUM**:

```python
# Few-shot evaluation
def evaluate_few_shot(model, dataset, shots_per_intent=5):
    """
    Test if model can learn intents from very few examples
    """
    intents = dataset['intent'].unique()
    results = {}

    for intent in intents:
        # Take only N examples for this intent
        intent_data = dataset[dataset['intent'] == intent]
        train_examples = intent_data.sample(n=shots_per_intent)
        test_examples = intent_data.drop(train_examples.index)

        # Fine-tune on few-shot examples (or just test if using pretrained)
        # ... training code

        # Evaluate on held-out examples of same intent
        accuracy = evaluate(model, test_examples)
        results[intent] = accuracy

    return np.mean(list(results.values()))

# Calibration evaluation
from sklearn.calibration import calibration_curve

def evaluate_calibration(model, test_loader):
    all_probs = []
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)

            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # Compute Expected Calibration Error
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    correct = (all_preds == all_labels)

    # Bin predictions by confidence
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(all_probs, bins) - 1

    ece = 0
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        if bin_mask.sum() > 0:
            bin_confidence = all_probs[bin_mask].mean()
            bin_accuracy = correct[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(all_probs)

            ece += bin_weight * abs(bin_confidence - bin_accuracy)

    print(f"Expected Calibration Error: {ece:.3f}")
    print(f"Average confidence: {all_probs.mean():.3f}")
    print(f"Accuracy: {correct.mean():.3f}")

    # Plot reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct, all_probs, n_bins=10
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig('calibration_plot.png')

    return ece

# Hard negative test set creation
def create_adversarial_test_set(dataset, intent_keywords):
    """
    Create test queries with maximum keyword overlap with wrong intents
    """
    adversarial_examples = []

    for intent, keywords in intent_keywords.items():
        # Find other intents with shared keywords
        for other_intent, other_keywords in intent_keywords.items():
            if intent != other_intent:
                shared = set(keywords) & set(other_keywords)
                if len(shared) >= 3:  # Significant overlap
                    # Create query using shared keywords but correct intent is 'intent'
                    query = f"{' '.join(list(shared)[:3])} context_word_for_{intent}"
                    adversarial_examples.append({
                        'query': query,
                        'intent': intent,
                        'confounding_intent': other_intent,
                        'shared_keywords': list(shared)
                    })

    return adversarial_examples
```

#### Expected Impact
- **Reveals true performance**: Few-shot evaluation likely shows **65-75%** instead of 98%
- **Calibration**: ECE probably high (0.15-0.25), showing overconfidence
- **Development guidance**: Identifies which intents are memorized vs understood

**This is not a fix but a diagnostic**. Use these metrics to:
1. Understand the real scope of your problem
2. Track whether interventions actually improve generalization (not just test accuracy)
3. Decide when the model is "good enough"

**Sources**:
- [Exploring Zero and Few-shot Techniques for Intent Classification](https://arxiv.org/abs/2305.07157)
- [A review on NLP zero-shot and few-shot learning](https://link.springer.com/article/10.1007/s42452-025-07225-5)
- [Calibrating Neural Networks](https://geoffpleiss.com/blog/nn_calibration.html)

---

### 8. Additional Powerful Techniques

#### 8.1 Mixup and Manifold Mixup

**How It Works**:
Traditional Mixup interpolates between training examples:
```
x_mixed = λ * x_i + (1 - λ) * x_j
y_mixed = λ * y_i + (1 - λ) * y_j
```

Manifold Mixup does this in hidden layers, creating "out-of-manifold" examples.

**Nonlinear Mixup** (for text): Uses learned nonlinear interpolation for both inputs and labels.

**Relevance**: MEDIUM - Helps with overfitting, but implementation for text is non-trivial. Better suited for vision tasks.

**Implementation**:
```python
class ManifoldMixupTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get hidden states from a random layer
        layer_idx = np.random.randint(0, model.config.num_hidden_layers)

        # Forward pass with hooks to capture hidden states
        hidden_states = []
        def hook(module, input, output):
            hidden_states.append(output[0])

        handle = model.distilbert.transformer.layer[layer_idx].register_forward_hook(hook)

        # First forward to get hidden states
        _ = model(**inputs)
        h = hidden_states[0]

        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices for mixing
        batch_size = h.size(0)
        index = torch.randperm(batch_size).to(h.device)

        # Mix hidden states
        mixed_h = lam * h + (1 - lam) * h[index]

        # Continue forward from mixed hidden state
        # (requires model modification to accept intermediate hidden states)
        # ... implementation details

        handle.remove()
```

**Expected Impact**: **90-93%** accuracy with smoother decision boundaries.

**Sources**:
- [Nonlinear Mixup: Out-Of-Manifold Data Augmentation](https://aaai.org/papers/04044-nonlinear-mixup-out-of-manifold-data-augmentation-for-text-classification/)
- [Manifold Mixup improves text recognition](https://www.arxiv-vanity.com/papers/1903.04246/)

---

#### 8.2 Focal Loss for Intent Classification

**How It Works**:
Standard cross-entropy treats all examples equally. Focal Loss down-weights easy examples:
```
FL(p_t) = -(1 - p_t)^γ * log(p_t)
```

Where:
- p_t is the predicted probability for the correct class
- γ (typically 2) controls how much to down-weight easy examples
- When model is confident (p_t ≈ 1), loss ≈ 0
- When model is uncertain (p_t ≈ 0.5), loss remains high

**Relevance**: **MEDIUM-HIGH** - Forces model to focus on hard examples (your hard negatives) instead of easy template matches.

**Implementation**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights, if imbalanced
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class

        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Usage
criterion = FocalLoss(gamma=2.0)
loss = criterion(logits, labels)
```

**Expected Impact**:
- Focus on hard negatives improves discrimination
- **87-92%** accuracy with better performance on confusing intent pairs
- Works well combined with curriculum learning

**Sources**:
- [Focal Loss: Class Imbalance in Detection](https://www.ultralytics.com/glossary/focal-loss)
- [Multi-class classification with focal loss](https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/)

---

#### 8.3 Adversarial Training

**How It Works**:
Generate adversarial examples by adding perturbations that maximally increase loss:
```
x_adv = x + ε * sign(∇_x L(x, y))
```

Train on both original and adversarial examples to learn robust features.

For text: Use synonym substitution or embedding perturbations instead of pixel changes.

**Relevance**: **MEDIUM** - Can improve robustness, but may not address template memorization directly. Better for handling noisy real-world inputs.

**Implementation**:
```python
class AdversarialTrainer(Trainer):
    def __init__(self, *args, epsilon=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Compute gradients w.r.t. embeddings
        embeds = model.get_input_embeddings()(inputs['input_ids'])
        embeds.retain_grad()

        loss.backward(retain_graph=True)

        # Generate adversarial perturbation
        grad = embeds.grad
        perturbation = self.epsilon * grad.sign()

        # Forward pass with perturbed embeddings
        adv_embeds = embeds + perturbation
        # (requires model modification to accept embeddings directly)
        adv_outputs = model(inputs_embeds=adv_embeds.detach())

        # Combined loss
        total_loss = (loss + adv_outputs.loss) / 2

        return (total_loss, outputs) if return_outputs else total_loss
```

**Expected Impact**: **86-90%** accuracy with better robustness to input variations.

**Sources**:
- [Adversarial Training for Text Classification](https://www.sciencedirect.com/science/article/pii/S1319157823002513)
- [Impact of Adversarial Training on Robustness and Generalizability](https://arxiv.org/html/2211.05523v3)

---

#### 8.4 Temperature Scaling (Post-Training Calibration)

**How It Works**:
After training, find a temperature T that calibrates the model's confidence:
```
p_calibrated = softmax(logits / T)
```

Doesn't retrain weights, just rescales outputs. Takes milliseconds.

**Relevance**: **MEDIUM** - Won't change accuracy but will fix overconfidence. Useful for deployment even if you use other techniques.

**Implementation**:
```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels):
        """Find optimal temperature on validation set"""
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self.temperature.item()

# Usage after training
temp_scaler = TemperatureScaling()
val_logits, val_labels = get_validation_predictions(model, val_loader)
optimal_temp = temp_scaler.calibrate(val_logits, val_labels)

# At inference
calibrated_logits = logits / optimal_temp
probs = F.softmax(calibrated_logits, dim=-1)
```

**Expected Impact**:
- Accuracy unchanged (98%)
- Confidence properly calibrated (ECE drops from ~0.20 to ~0.05)
- Better for production confidence thresholds

**Sources**:
- [Calibrating Neural Networks](https://geoffpleiss.com/blog/nn_calibration.html)
- [Temperature Scaling GitHub](https://github.com/gpleiss/temperature_scaling)

---

## Recommended Implementation Strategy

### Phase 1: Quick Wins (1-2 days)
**Goal**: Achieve 85-90% accuracy with minimal code changes

1. **Label Smoothing** (ε=0.1)
   - Built into Hugging Face Trainer
   - Expected: 88-93% accuracy

2. **Increased Dropout** (0.2-0.3)
   - One-line change in model config
   - Expected: 89-93% accuracy

3. **Better Evaluation Metrics**
   - Implement ECE, few-shot evaluation
   - Understand true performance gap

**Expected Combined Impact**: **86-90%** accuracy with better calibration

---

### Phase 2: Moderate Effort (3-5 days)
**Goal**: Robust learning with semantic understanding

4. **Curriculum Learning**
   - Implement static curriculum based on keyword overlap
   - Start with distinct intents, gradually add hard negatives
   - Expected: 85-89% with stabler training

5. **R-Drop Regularization**
   - Add consistency loss between dropout variations
   - Expected: 87-91% with better generalization

6. **Focal Loss** (γ=2.0)
   - Replace cross-entropy to focus on hard examples
   - Expected: 87-92% with better hard negative performance

**Expected Combined Impact**: **85-88%** accuracy with significantly better real-world generalization

---

### Phase 3: Advanced Techniques (1-2 weeks)
**Goal**: State-of-the-art intent understanding

7. **Contrastive Learning**
   - Implement supervised contrastive loss
   - Add projection head to model
   - Expected: 85-90% with excellent semantic clustering

8. **Indonesian Paraphrasing**
   - Fine-tune mBART50 on ParaCotta
   - Generate 2-3 paraphrases per training example
   - Expected: 5-10% improvement on real queries

9. **Multi-Granularity Hard Negative Curriculum**
   - Generate synthetic hard negatives at multiple difficulty levels
   - Implement coarse-to-fine training
   - Expected: 24% improvement on hardest cases (research showed this)

**Expected Combined Impact**: **83-87%** accuracy with robust generalization to unseen queries

---

### Phase 4: Production Optimization (2-3 days)

10. **Temperature Scaling**
    - Calibrate confidence scores post-training
    - Essential for production confidence thresholds

11. **Adversarial Test Set Creation**
    - Build test set with maximum keyword overlap
    - Continuous evaluation benchmark

---

## Key Academic Papers

### Most Relevant to Your Problem

1. **"Learning label smoothing for text classification"** (2024)
   - [PeerJ Link](https://peerj.com/articles/cs-2005/)
   - Shows 5% accuracy improvement with adaptive label smoothing on text classification

2. **"Multi-Granularity Hard-Negative Synthesis"** (2024)
   - [ArXiv Link](https://arxiv.org/html/2509.00842)
   - Curriculum learning with synthetic hard negatives at multiple difficulty levels
   - 24% improvement on difficult benchmarks

3. **"Few-shot intent detection with mutual information and contrastive learning"** (2024)
   - [ScienceDirect Link](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011128)
   - Specifically designed for intent classification with hard negatives

4. **"Improving Imbalanced Text Classification with Dynamic Curriculum Learning"** (2022)
   - [ArXiv Link](https://arxiv.org/abs/2210.14724)
   - Self-paced dynamic curriculum learning (SPDCL) for text classification

5. **"Study of Text Augmentation with Paraphrasing for Indonesian"** (2024)
   - [Journal Link](https://join.if.uinsgd.ac.id/index.php/join/article/view/1472)
   - Fine-tuned mBART50 significantly improves F1 for Indonesian classification

### Foundational Papers

6. **"Rethinking Calibration of Deep Neural Networks"** (2021)
   - [NeurIPS Link](https://proceedings.neurips.cc/paper/2021/hash/61f3a6dbc9120ea78ef75544826c814e-Abstract.html)
   - Shows overconfidence may not hurt if post-hoc calibration is used

7. **"Best Practices and Lessons Learned on Synthetic Data for Language Models"** (2024)
   - [ArXiv Link](https://arxiv.org/html/2404.07503v1)
   - Comprehensive overview of synthetic data limitations and solutions

8. **"Training and Evaluating Language Models with Template-based Data Generation"** (2024)
   - [ArXiv Link](https://arxiv.org/abs/2411.18104)
   - Directly addresses your template-based training scenario

9. **"Nonlinear Mixup: Out-Of-Manifold Data Augmentation for Text Classification"** (AAAI)
   - [AAAI Link](https://aaai.org/papers/04044-nonlinear-mixup-out-of-manifold-data-augmentation-for-text-classification/)
   - Mixup regularization specifically for text

10. **"Impact of Adversarial Training on Robustness and Generalizability of Language Models"** (2023)
    - [ArXiv Link](https://arxiv.org/html/2211.05523v3)
    - Shows PGD adversarial training improves generalization for text (unlike vision)

---

## Conclusion

Your 98% accuracy is a **red flag, not a success metric**. The model is exploiting distributional shortcuts in your synthetic data rather than learning semantic intent understanding.

### Top 3 Recommendations for 85-90% Target:

1. **Label Smoothing (ε=0.1) + Increased Dropout (0.2-0.3)**
   - Easiest to implement (< 1 day)
   - Expected: 86-90% accuracy
   - Prevents overconfident memorization

2. **Curriculum Learning + Focal Loss**
   - Moderate effort (3-5 days)
   - Expected: 85-88% accuracy
   - Forces focus on hard negatives, prevents early shortcut learning

3. **Supervised Contrastive Learning**
   - Higher effort (1 week)
   - Expected: 85-90% accuracy
   - Best semantic understanding, robust to template variations

### Critical Success Metrics:

Don't just track test accuracy. Monitor:
- **Expected Calibration Error (ECE)**: Should be < 0.10
- **Few-shot accuracy** (5 examples per intent): Should be > 70%
- **Hard negative accuracy** (queries with 3+ shared keywords): Should be > 75%
- **Confidence on errors**: Should be < 0.6 (model should be uncertain when wrong)

If you achieve 87% accuracy with ECE < 0.10 and few-shot accuracy > 70%, you have a model that **understands intents** rather than memorizes templates.

---

## All Sources

### Label Smoothing
- [Learning label smoothing for text classification - PeerJ](https://peerj.com/articles/cs-2005/)
- [Label Smoothing for Enhanced Text Sentiment Classification](https://arxiv.org/html/2312.06522v1)
- [Understanding Why Label Smoothing Degrades Selective Classification](https://arxiv.org/html/2403.14715v1)

### Contrastive Learning
- [Few-shot intent detection with mutual information and contrastive learning](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011128)
- [Contrastive Learning in NLP](https://medium.com/data-scientists-diary/contrastive-learning-in-nlp-dc0e1a5bb23c)
- [The Beginner's Guide to Contrastive Learning](https://www.v7labs.com/blog/contrastive-learning-guide)

### Curriculum Learning
- [Multi-Granularity Hard-Negative Synthesis](https://arxiv.org/html/2509.00842)
- [Curriculum DPO on Synthetic Negatives](https://arxiv.org/abs/2505.17558)
- [Dynamic Curriculum Learning for Imbalanced Text Classification](https://arxiv.org/abs/2210.14724)
- [Contrastive learning with hard negatives for sentence embeddings](https://www.sciencedirect.com/science/article/abs/pii/S1568494625009962)

### Data Augmentation
- [Study of Text Augmentation with Paraphrasing for Indonesian](https://join.if.uinsgd.ac.id/index.php/join/article/view/1472)
- [Data Augmentation by Backtranslation](https://github.com/vietai/dab)
- [IndoNLG: Indonesian NLG Benchmark](https://github.com/IndoNLP/indonlg)
- [Backtranslation and paraphrasing in the LLM era](https://arxiv.org/abs/2507.14590)

### Regularization
- [AttentionDrop: Novel Regularization for Transformers](https://arxiv.org/html/2504.12088)
- [Advanced Regularization Protocols for Transformers](https://medium.com/@hassanbinabid/the-art-and-science-of-hyperparameter-optimization-in-llm-fine-tuning-f95bc6e9a80b)
- [Self-knowledge distillation via dropout](https://www.sciencedirect.com/science/article/abs/pii/S1077314223001005)
- [Evaluation of Regularization Techniques for Transformers](https://link.springer.com/chapter/10.1007/978-3-031-36616-1_25)

### Alternative Architectures
- [Alternative non-BERT choices for low-resource languages](https://aclanthology.org/2022.deeplo-1.20/)
- [Limitations of Transformers on Clinical Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8387496/)

### Few-Shot and Zero-Shot Evaluation
- [Exploring Zero and Few-shot Techniques for Intent Classification](https://arxiv.org/abs/2305.07157)
- [A review on NLP zero-shot and few-shot learning](https://link.springer.com/article/10.1007/s42452-025-07225-5)
- [A Generalization Theory for Zero-Shot Prediction](https://arxiv.org/html/2507.09128)

### Synthetic Data Limitations
- [Best Practices and Lessons Learned on Synthetic Data](https://arxiv.org/html/2404.07503v1)
- [Training with Template-based Data Generation](https://arxiv.org/abs/2411.18104)
- [Synthetic Data Generation with Large Language Models](https://arxiv.org/html/2503.14023v1)
- [Synthetic Data for Text Classification: Potential and Limitations](https://openreview.net/forum?id=MmBjKmHIND)

### Mixup Techniques
- [Nonlinear Mixup for Text Classification](https://aaai.org/papers/04044-nonlinear-mixup-out-of-manifold-data-augmentation-for-text-classification/)
- [Manifold Mixup improves text recognition](https://www.arxiv-vanity.com/papers/1903.04246/)
- [A Survey on Mixup Augmentations and Beyond](https://arxiv.org/html/2409.05202v1)

### Focal Loss
- [Focal Loss: Class Imbalance in Detection](https://www.ultralytics.com/glossary/focal-loss)
- [Multi-class classification with focal loss](https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/)
- [Focal loss for handling class imbalance](https://medium.com/data-science-ecom-express/focal-loss-for-handling-the-issue-of-class-imbalance-be7addebd856)

### Calibration
- [Calibrating Neural Networks](https://geoffpleiss.com/blog/nn_calibration.html)
- [Temperature Scaling GitHub](https://github.com/gpleiss/temperature_scaling)
- [Rethinking Calibration of Deep Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/61f3a6dbc9120ea78ef75544826c814e-Abstract.html)
- [Adaptive Temperature Scaling](https://link.springer.com/article/10.1007/s00521-024-09505-4)

### Adversarial Training
- [Adversarial Training for Text Classification](https://www.sciencedirect.com/science/article/pii/S1319157823002513)
- [Impact of Adversarial Training on Robustness and Generalizability](https://arxiv.org/html/2211.05523v3)
- [Regularizing Hard Examples Improves Adversarial Training](http://www.jmlr.org/papers/volume26/22-1428/22-1428.pdf)

### R-Drop and Self-Distillation
- [Self-knowledge distillation via dropout](https://www.sciencedirect.com/science/article/abs/pii/S1077314223001005)
- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- [Knowledge distillation for LLMs](https://link.springer.com/article/10.1007/s10462-025-11423-3)

### Indonesian Models
- [IndoNLG Repository](https://github.com/IndoNLP/indonlg)
- [IndoBART-v2](https://huggingface.co/indobenchmark/indobart-v2)
- [Enhancing Question Generation in Bahasa](https://www.iieta.org/journals/ria/paper/10.18280/ria.380421)
