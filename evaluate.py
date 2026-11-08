import argparse
import json
import os
from pathlib import Path

from retrieval import DocumentIndex


DEFAULT_CASES = Path(__file__).resolve().parent / "evaluation" / "cases.json"


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON array")

    for number, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {number} must be an object")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Evaluation case {number} needs a non-empty question")
        for key in ("expected_sources", "expected_terms"):
            values = case.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ValueError(f"Evaluation case {number} needs a {key} string array")
        if not isinstance(case.get("expect_no_results", False), bool):
            raise ValueError(f"Evaluation case {number} expect_no_results must be boolean")
    return cases


def evaluate(index: DocumentIndex, cases: list[dict]) -> int:
    failures = 0
    for number, case in enumerate(cases, start=1):
        matches = index.search(case["question"])
        retrieved_sources = {match["path"] for match in matches}
        retrieved_text = "\n".join(match["content"] for match in matches).casefold()
        issues = []

        missing_sources = set(case["expected_sources"]) - retrieved_sources
        if missing_sources:
            issues.append(f"missing sources: {', '.join(sorted(missing_sources))}")

        missing_terms = [
            term for term in case["expected_terms"] if term.casefold() not in retrieved_text
        ]
        if missing_terms:
            issues.append(f"missing expected facts: {', '.join(missing_terms)}")

        if case.get("expect_no_results") and matches:
            issues.append(
                "expected no results, got: "
                + ", ".join(sorted(retrieved_sources))
            )

        status = "FAIL" if issues else "PASS"
        print(f"{status} {number}. {case['question']}")
        for issue in issues:
            print(f"  - {issue}")
        failures += bool(issues)

    print(f"\n{len(cases) - failures}/{len(cases)} evaluation cases passed")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check workload retrieval against expected sources and facts."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path(os.getenv("WORKLOAD_DOCS_PATH", "workloads")),
        help="Workload documents directory (default: WORKLOAD_DOCS_PATH or workloads/)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Evaluation cases JSON file",
    )
    args = parser.parse_args()

    try:
        cases = load_cases(args.cases)
        index = DocumentIndex(args.docs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    try:
        return evaluate(index, cases)
    finally:
        index.close()


if __name__ == "__main__":
    raise SystemExit(main())
