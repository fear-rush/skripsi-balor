#!/usr/bin/env python3
"""
Evaluation script for LLM Slot Extractor.

Usage:
    # Make sure llama-server is running first:
    # ./scripts/start_llm_server.sh

    # Run evaluation:
    python scripts/evaluate_llm_extractor.py

    # With custom server URL:
    python scripts/evaluate_llm_extractor.py --server http://localhost:8080
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_slot_extractor import SlotExtractor


def load_eval_dataset(path: str) -> Dict[str, Any]:
    """Load evaluation dataset from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_task(expected: str, predicted: str) -> bool:
    """Compare task labels."""
    return expected.lower() == predicted.lower()


def compare_entities(expected: Dict, predicted: Dict) -> Tuple[float, Dict]:
    """
    Compare entity extraction.
    Returns (score, details) where score is 0.0-1.0.
    """
    if not expected and not predicted:
        return 1.0, {"match": "both_empty"}

    if not expected:
        return 0.5, {"match": "expected_empty_but_predicted", "predicted": predicted}

    if not predicted:
        return 0.0, {"match": "predicted_empty", "expected": expected}

    # Count matches
    matches = 0
    total = len(expected)
    details = {"expected": expected, "predicted": predicted, "matches": []}

    for key, exp_val in expected.items():
        if exp_val is None:
            total -= 1
            continue

        pred_val = predicted.get(key)
        if pred_val is not None:
            # Flexible string matching
            exp_str = str(exp_val).lower()
            pred_str = str(pred_val).lower()

            if exp_str == pred_str or exp_str in pred_str or pred_str in exp_str:
                matches += 1
                details["matches"].append(key)

    score = matches / total if total > 0 else 1.0
    return score, details


def compare_multi_intent(expected: List[str], predicted: List[str]) -> Tuple[float, Dict]:
    """Compare multi-intent detection."""
    if not expected and not predicted:
        return 1.0, {"match": "both_empty"}

    if not expected:
        # Predicted multi-intent when none expected - partial credit
        return 0.5, {"match": "unexpected_multi_intent", "predicted": predicted}

    if not predicted:
        return 0.0, {"match": "missed_multi_intent", "expected": expected}

    # Count matches
    exp_set = set(e.lower() for e in expected)
    pred_set = set(p.lower() for p in predicted)

    matches = len(exp_set & pred_set)
    total = len(exp_set)

    score = matches / total if total > 0 else 1.0
    return score, {"expected": expected, "predicted": predicted, "matches": list(exp_set & pred_set)}


def evaluate_single(
    extractor: SlotExtractor,
    query: str,
    expected: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate a single query."""
    start_time = time.time()
    result = extractor.extract(query)
    latency = time.time() - start_time

    # Compare task
    task_match = compare_task(
        expected.get("task", ""),
        result.task
    )

    # Compare entities
    entity_score, entity_details = compare_entities(
        expected.get("entities", {}),
        result.entities
    )

    # Compare multi-intent
    multi_intent_score, multi_intent_details = compare_multi_intent(
        expected.get("multi_intent", []),
        result.multi_intent
    )

    # Compare needs_clarification
    clarification_match = (
        expected.get("needs_clarification", False) == result.needs_clarification
    )

    return {
        "query": query,
        "latency": latency,
        "task_match": task_match,
        "entity_score": entity_score,
        "entity_details": entity_details,
        "multi_intent_score": multi_intent_score,
        "multi_intent_details": multi_intent_details,
        "clarification_match": clarification_match,
        "expected": expected,
        "predicted": result.to_dict(),
        "error": result.error
    }


def run_evaluation(
    extractor: SlotExtractor,
    dataset: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """Run full evaluation on dataset."""
    queries = dataset["queries"]
    results = []

    # Metrics
    task_correct = 0
    entity_scores = []
    multi_intent_scores = []
    clarification_correct = 0
    latencies = []
    errors = 0

    # By category
    category_metrics = {}

    if verbose:
        print(f"Evaluating {len(queries)} queries...")
        print("=" * 60)

    for i, item in enumerate(queries):
        query = item["query"]
        expected = item["expected"]
        category = item.get("category", "unknown")

        if verbose:
            print(f"\n[{i+1}/{len(queries)}] {query}")

        eval_result = evaluate_single(extractor, query, expected)
        results.append(eval_result)

        # Update metrics
        if eval_result["error"]:
            errors += 1
            if verbose:
                print(f"  ERROR: {eval_result['error']}")
            continue

        if eval_result["task_match"]:
            task_correct += 1

        entity_scores.append(eval_result["entity_score"])
        multi_intent_scores.append(eval_result["multi_intent_score"])

        if eval_result["clarification_match"]:
            clarification_correct += 1

        latencies.append(eval_result["latency"])

        # Category metrics
        if category not in category_metrics:
            category_metrics[category] = {"total": 0, "task_correct": 0}
        category_metrics[category]["total"] += 1
        if eval_result["task_match"]:
            category_metrics[category]["task_correct"] += 1

        if verbose:
            pred = eval_result["predicted"]
            status = "OK" if eval_result["task_match"] else "WRONG"
            print(f"  Task: {pred['task']} [{status}]")
            print(f"  Entities: {pred['entities']}")
            if pred.get("multi_intent"):
                print(f"  Multi-intent: {pred['multi_intent']}")
            print(f"  Latency: {eval_result['latency']:.2f}s")

    # Calculate summary metrics
    total = len(queries)
    valid = total - errors

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "valid_queries": valid,
        "errors": errors,
        "metrics": {
            "task_accuracy": task_correct / valid if valid > 0 else 0,
            "entity_score_mean": sum(entity_scores) / len(entity_scores) if entity_scores else 0,
            "multi_intent_score_mean": sum(multi_intent_scores) / len(multi_intent_scores) if multi_intent_scores else 0,
            "clarification_accuracy": clarification_correct / valid if valid > 0 else 0,
        },
        "latency": {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
        },
        "by_category": {
            cat: {
                "accuracy": m["task_correct"] / m["total"] if m["total"] > 0 else 0,
                "total": m["total"],
                "correct": m["task_correct"]
            }
            for cat, m in category_metrics.items()
        },
        "results": results
    }

    return summary


def print_summary(summary: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal Queries: {summary['total_queries']}")
    print(f"Valid Queries: {summary['valid_queries']}")
    print(f"Errors: {summary['errors']}")

    print("\n--- Metrics ---")
    metrics = summary["metrics"]
    print(f"Task Accuracy: {metrics['task_accuracy']:.1%}")
    print(f"Entity Score: {metrics['entity_score_mean']:.1%}")
    print(f"Multi-Intent Score: {metrics['multi_intent_score_mean']:.1%}")
    print(f"Clarification Accuracy: {metrics['clarification_accuracy']:.1%}")

    print("\n--- Latency ---")
    lat = summary["latency"]
    print(f"Mean: {lat['mean']:.3f}s")
    print(f"Min: {lat['min']:.3f}s")
    print(f"Max: {lat['max']:.3f}s")

    print("\n--- By Category ---")
    for cat, m in summary["by_category"].items():
        print(f"  {cat}: {m['accuracy']:.1%} ({m['correct']}/{m['total']})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Slot Extractor")
    parser.add_argument(
        "--server", "-s",
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
        default="evaluation/llm_slot_extraction_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    # Check server
    extractor = SlotExtractor(server_url=args.server)
    if not extractor.is_server_running():
        print("ERROR: llama-server is not running!")
        print("")
        print("Start the server first:")
        print("  ./scripts/start_llm_server.sh")
        print("")
        return 1

    print(f"Server: {args.server}")
    print(f"Dataset: {args.dataset}")

    # Load dataset
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / args.dataset

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1

    dataset = load_eval_dataset(dataset_path)
    print(f"Loaded {len(dataset['queries'])} queries")

    # Run evaluation
    summary = run_evaluation(extractor, dataset, verbose=not args.quiet)

    # Print summary
    print_summary(summary)

    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
