from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate
from retrieval import DocumentIndex


class RetrievalEvaluationTests(unittest.TestCase):
    def test_expected_source_facts_and_no_result_case_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.md").write_text(
                "The worker listens on port 4317.\n", encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)
            cases = [
                {
                    "question": "Which port does the worker listen on?",
                    "expected_sources": ["service.md"],
                    "expected_terms": ["4317"],
                },
                {
                    "question": "What is the quasar telemetry identifier 9183?",
                    "expected_sources": [],
                    "expected_terms": [],
                    "expect_no_results": True,
                },
            ]

            with redirect_stdout(StringIO()):
                failures = evaluate(index, cases)

            self.assertEqual(failures, 0)

    def test_missing_expected_fact_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.md").write_text(
                "The worker listens on port 4317.\n", encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)
            cases = [
                {
                    "question": "Which port does the worker listen on?",
                    "expected_sources": ["service.md"],
                    "expected_terms": ["5192"],
                }
            ]

            with redirect_stdout(StringIO()):
                failures = evaluate(index, cases)

            self.assertEqual(failures, 1)


if __name__ == "__main__":
    unittest.main()
