"""
Automated Chatbot Test Runner

Tests the LLM-based e-commerce chatbot with single-turn and multi-turn test cases.
Uses direct module testing (not HTTP API) as specified.

Usage:
    uv run python scripts/test_chatbot_automated.py

    # Run specific test category
    uv run python scripts/test_chatbot_automated.py --single-turn
    uv run python scripts/test_chatbot_automated.py --multi-turn
"""

import json
import sys
import os
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import LLMChatbot
from src.conversation_context import ConversationContext, ContextStore
from src.config import LLM_SERVER_URL, DB_CONFIG, WHATSAPP_NUMBER


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    test_name: str
    passed: bool
    expected_task: str
    actual_task: str
    expected_entities: Dict[str, Any]
    actual_entities: Dict[str, Any]
    response: str
    latency_ms: float
    errors: List[str]
    confidence: float
    turn_number: int = 1


@dataclass
class TestSummary:
    """Summary of test run."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    single_turn_results: List[TestResult]
    multi_turn_results: List[Dict]
    error_results: List[TestResult]


class ChatbotTestRunner:
    """
    Automated test runner for the chatbot.

    Tests the chatbot directly without going through HTTP API.
    """

    def __init__(self, test_cases_path: str = "evaluation/chatbot_test_cases.json"):
        self.test_cases_path = test_cases_path
        self.chatbot = None
        self.results: List[TestResult] = []
        self.multi_turn_results: List[Dict] = []
        self.error_results: List[TestResult] = []

        # Load test cases
        with open(test_cases_path, 'r') as f:
            self.test_cases = json.load(f)

    def _init_chatbot(self):
        """Initialize chatbot if not already initialized."""
        if self.chatbot is None:
            print("Initializing chatbot...")
            self.chatbot = LLMChatbot(
                llm_server_url=LLM_SERVER_URL,
                db_config=DB_CONFIG,
                whatsapp_number=WHATSAPP_NUMBER
            )

            if not self.chatbot.is_llm_available():
                raise RuntimeError("LLM server is not running. Start with ./scripts/start_llm_server.sh")

            print("Chatbot initialized successfully")

    def run_single_turn_test(self, test_case: Dict) -> TestResult:
        """Run a single-turn test case."""
        test_id = test_case['id']
        test_name = test_case['name']
        query = test_case['query']
        expected_task = test_case['expected_task']
        expected_entities = test_case.get('expected_entities', {})
        expected_response_contains = test_case.get('expected_response_contains', [])
        expected_multi_intent = test_case.get('expected_multi_intent', [])

        errors = []

        # Process query
        start_time = time.time()
        result = self.chatbot.process(query)
        latency = (time.time() - start_time) * 1000

        actual_task = result.task
        actual_entities = result.entities
        response = result.response

        # Check task
        if isinstance(expected_task, list):
            # Allow multiple valid tasks
            if actual_task not in expected_task:
                errors.append(f"Task mismatch: expected one of {expected_task}, got {actual_task}")
        else:
            if actual_task != expected_task:
                errors.append(f"Task mismatch: expected {expected_task}, got {actual_task}")

        # Check entities
        for key, expected_value in expected_entities.items():
            actual_value = actual_entities.get(key)
            if str(actual_value).lower() != str(expected_value).lower():
                errors.append(f"Entity mismatch for {key}: expected {expected_value}, got {actual_value}")

        # Check response contains
        response_lower = response.lower()
        for expected_text in expected_response_contains:
            if expected_text.lower() not in response_lower:
                errors.append(f"Response missing: '{expected_text}'")

        # Check multi-intent
        if expected_multi_intent:
            for intent in expected_multi_intent:
                if intent not in result.multi_intent:
                    errors.append(f"Missing multi-intent: {intent}")

        passed = len(errors) == 0

        return TestResult(
            test_id=test_id,
            test_name=test_name,
            passed=passed,
            expected_task=str(expected_task),
            actual_task=actual_task,
            expected_entities=expected_entities,
            actual_entities=actual_entities,
            response=response[:200] + "..." if len(response) > 200 else response,
            latency_ms=round(latency, 1),
            errors=errors,
            confidence=result.confidence
        )

    def run_multi_turn_test(self, test_case: Dict) -> Dict:
        """Run a multi-turn conversation test."""
        test_id = test_case['id']
        test_name = test_case['name']
        turns = test_case['turns']

        # Create fresh context for this conversation
        context = ConversationContext(f"test-{test_id}")

        turn_results = []
        all_passed = True

        for i, turn in enumerate(turns, 1):
            user_query = turn['user']
            expect_clarification = turn.get('expect_clarification', False)
            expected_task = turn.get('expected_task')
            expected_entities = turn.get('expected_entities', {})
            expected_response_contains = turn.get('expected_response_contains', [])

            errors = []

            # Process with context
            start_time = time.time()
            result = self.chatbot.process_with_context(user_query, context)
            latency = (time.time() - start_time) * 1000

            # Check clarification expectation
            if expect_clarification != result.needs_clarification:
                errors.append(f"Clarification mismatch: expected {expect_clarification}, got {result.needs_clarification}")

            # Check task (if not expecting clarification or if specified)
            if expected_task and result.task != expected_task:
                errors.append(f"Task mismatch: expected {expected_task}, got {result.task}")

            # Check entities
            for key, expected_value in expected_entities.items():
                actual_value = result.entities.get(key)
                if actual_value is None:
                    errors.append(f"Missing entity: {key}")
                elif str(actual_value).lower() != str(expected_value).lower():
                    errors.append(f"Entity mismatch for {key}: expected {expected_value}, got {actual_value}")

            # Check response contains
            response_lower = result.response.lower()
            for expected_text in expected_response_contains:
                if expected_text.lower() not in response_lower:
                    errors.append(f"Response missing: '{expected_text}'")

            passed = len(errors) == 0
            if not passed:
                all_passed = False

            turn_results.append({
                "turn": i,
                "query": user_query,
                "passed": passed,
                "task": result.task,
                "entities": result.entities,
                "response": result.response[:150] + "..." if len(result.response) > 150 else result.response,
                "needs_clarification": result.needs_clarification,
                "latency_ms": round(latency, 1),
                "confidence": result.confidence,
                "errors": errors
            })

        return {
            "test_id": test_id,
            "test_name": test_name,
            "passed": all_passed,
            "turns": turn_results
        }

    def run_all_tests(self, run_single=True, run_multi=True, run_error=True) -> TestSummary:
        """Run all test cases."""
        self._init_chatbot()

        print("\n" + "=" * 70)
        print("AUTOMATED CHATBOT TESTING")
        print("=" * 70)

        self.results = []
        self.multi_turn_results = []
        self.error_results = []

        # Single-turn tests
        if run_single:
            print(f"\n1. SINGLE-TURN TESTS ({len(self.test_cases['single_turn_tests'])} tests)")
            print("-" * 40)

            for test in self.test_cases['single_turn_tests']:
                result = self.run_single_turn_test(test)
                self.results.append(result)

                status = "PASS" if result.passed else "FAIL"
                print(f"  [{status}] {result.test_id}: {result.test_name} ({result.latency_ms}ms)")
                if not result.passed:
                    for error in result.errors:
                        print(f"         {error}")

        # Multi-turn tests
        if run_multi:
            print(f"\n2. MULTI-TURN TESTS ({len(self.test_cases['multi_turn_tests'])} conversations)")
            print("-" * 40)

            for test in self.test_cases['multi_turn_tests']:
                result = self.run_multi_turn_test(test)
                self.multi_turn_results.append(result)

                status = "PASS" if result['passed'] else "FAIL"
                print(f"  [{status}] {result['test_id']}: {result['test_name']}")
                for turn in result['turns']:
                    turn_status = "PASS" if turn['passed'] else "FAIL"
                    print(f"         Turn {turn['turn']}: [{turn_status}] {turn['query'][:30]}...")
                    if not turn['passed']:
                        for error in turn['errors']:
                            print(f"                   {error}")

        # Error handling tests
        if run_error:
            print(f"\n3. ERROR HANDLING TESTS ({len(self.test_cases['error_tests'])} tests)")
            print("-" * 40)

            for test in self.test_cases['error_tests']:
                result = self.run_single_turn_test(test)
                self.error_results.append(result)

                status = "PASS" if result.passed else "FAIL"
                print(f"  [{status}] {result.test_id}: {result.test_name}")

        # Calculate summary
        all_results = self.results + self.error_results
        all_latencies = [r.latency_ms for r in all_results]
        multi_turn_passed = sum(1 for r in self.multi_turn_results if r['passed'])

        single_passed = sum(1 for r in self.results if r.passed)
        error_passed = sum(1 for r in self.error_results if r.passed)
        total_passed = single_passed + error_passed + multi_turn_passed
        total_tests = len(self.results) + len(self.error_results) + len(self.multi_turn_results)

        summary = TestSummary(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            passed=total_passed,
            failed=total_tests - total_passed,
            pass_rate=round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0,
            avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0,
            single_turn_results=[asdict(r) for r in self.results],
            multi_turn_results=self.multi_turn_results,
            error_results=[asdict(r) for r in self.error_results]
        )

        return summary

    def print_summary(self, summary: TestSummary):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  Total Tests:     {summary.total_tests}")
        print(f"  Passed:          {summary.passed}")
        print(f"  Failed:          {summary.failed}")
        print(f"  Pass Rate:       {summary.pass_rate}%")
        print(f"  Avg Latency:     {summary.avg_latency_ms}ms")
        print("=" * 70)

        if summary.failed > 0:
            print("\nFAILED TESTS:")
            for result in summary.single_turn_results:
                if not result['passed']:
                    print(f"  - {result['test_id']}: {result['test_name']}")
                    for error in result['errors']:
                        print(f"      {error}")

            for result in summary.multi_turn_results:
                if not result['passed']:
                    print(f"  - {result['test_id']}: {result['test_name']}")
                    for turn in result['turns']:
                        if not turn['passed']:
                            print(f"      Turn {turn['turn']}: {turn['query'][:30]}")
                            for error in turn['errors']:
                                print(f"        {error}")

    def save_results(self, summary: TestSummary, output_path: str = None):
        """Save test results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation/test_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Automated Chatbot Test Runner")
    parser.add_argument('--single-turn', action='store_true', help="Run only single-turn tests")
    parser.add_argument('--multi-turn', action='store_true', help="Run only multi-turn tests")
    parser.add_argument('--save', action='store_true', help="Save results to JSON file")
    parser.add_argument('--output', type=str, help="Output file path for results")

    args = parser.parse_args()

    # Determine which tests to run
    run_single = True
    run_multi = True
    run_error = True

    if args.single_turn:
        run_single = True
        run_multi = False
        run_error = False
    elif args.multi_turn:
        run_single = False
        run_multi = True
        run_error = False

    # Run tests
    runner = ChatbotTestRunner()

    try:
        summary = runner.run_all_tests(
            run_single=run_single,
            run_multi=run_multi,
            run_error=run_error
        )

        runner.print_summary(summary)

        if args.save or args.output:
            runner.save_results(summary, args.output)

        # Exit with error code if tests failed
        if summary.failed > 0:
            sys.exit(1)

    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
