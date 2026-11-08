import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from retrieval import DocumentIndex
from ui import (
    SYSTEM_PROMPT,
    append_retrieved_sources,
    build_messages,
    safe_history,
)


class ChatHistoryTests(unittest.TestCase):
    def test_gradio_message_history_is_preserved(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        messages = build_messages(history, "How are you?")

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-3:], [*history, {"role": "user", "content": "How are you?"}])

    def test_malformed_history_items_are_ignored(self):
        self.assertEqual(
            safe_history([{"role": "system", "content": "ignored"}, None]),
            [],
        )

    def test_retrieved_context_includes_source_citation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.md").write_text(
                "The control plane listens on port 7419.\n", encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)

            with patch("ui.document_index", index):
                messages = build_messages([], "What port does the control plane use?")

            self.assertIn("[Source: service.md:1-1]", messages[1]["content"])
            self.assertIn("7419", messages[1]["content"])

    def test_application_appends_retrieved_source_citations(self):
        matches = [
            {"path": "service.md", "start_line": 2, "end_line": 4},
            {"path": "service.md", "start_line": 2, "end_line": 4},
        ]

        answer = append_retrieved_sources("The port is 7419.", matches)

        self.assertEqual(
            answer,
            "The port is 7419.\n\nRetrieved sources:\n- `service.md` (lines 2-4)",
        )

    def test_application_does_not_add_sources_when_retrieval_is_empty(self):
        self.assertEqual(append_retrieved_sources("No source found.", []), "No source found.")


if __name__ == "__main__":
    unittest.main()
