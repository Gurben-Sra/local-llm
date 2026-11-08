from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval import DocumentIndex, _chunks


class DocumentIndexTests(unittest.TestCase):
    def test_search_returns_source_path_and_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.md").write_text(
                "# Service\n\nThe worker listens on port 4317.\n", encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)

            results = index.search("Which port does the worker listen on?")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["path"], "service.md")
            self.assertEqual(results[0]["start_line"], 1)
            self.assertEqual(results[0]["end_line"], 3)
            self.assertIn("4317", results[0]["content"])

    def test_unsupported_and_hidden_files_are_not_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.md").write_text("visible deployment notes", encoding="utf-8")
            (root / ".hidden.md").write_text("hidden token secretvalue", encoding="utf-8")
            (root / "data.csv").write_text("unsupported token secretvalue", encoding="utf-8")
            index = DocumentIndex(root)
            self.addCleanup(index.close)

            results = index.search("secretvalue")

            self.assertEqual(results, [])
            self.assertEqual(index.file_count, 1)

    def test_json_configuration_is_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "settings.json").write_text(
                '{"listener": {"port": 7419}}', encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)

            results = index.search("listener port")

            self.assertEqual(results[0]["path"], "settings.json")
            self.assertIn("7419", results[0]["content"])

    def test_long_lines_are_split_with_valid_line_references(self):
        long_text = "alpha " * 400

        chunks = list(_chunks(long_text))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(content) <= 1_600 for content, _, _ in chunks))
        self.assertTrue(all(start == end == 1 for _, start, end in chunks))

    def test_missing_directory_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(NotADirectoryError):
                DocumentIndex(missing)

    def test_search_is_safe_from_multiple_ui_worker_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.md").write_text(
                "The control plane listens on port 7419.\n", encoding="utf-8"
            )
            index = DocumentIndex(root)
            self.addCleanup(index.close)

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(index.search, ["control plane"] * 8))

            self.assertTrue(all(result[0]["path"] == "service.md" for result in results))

    def test_total_index_size_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("first document", encoding="utf-8")
            (root / "b.md").write_text("second document", encoding="utf-8")
            with patch("retrieval.MAX_TOTAL_BYTES", len("first document")):
                index = DocumentIndex(root)
            self.addCleanup(index.close)

            self.assertEqual(index.file_count, 1)
            self.assertEqual(index._indexed_bytes, len("first document"))

    def test_symlink_to_outside_root_is_not_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "workloads"
            root.mkdir()
            outside = base / "outside.md"
            outside.write_text("outsideonlytoken", encoding="utf-8")
            (root / "linked.md").symlink_to(outside)

            index = DocumentIndex(root)
            self.addCleanup(index.close)

            self.assertEqual(index.search("outsideonlytoken"), [])


if __name__ == "__main__":
    unittest.main()
