import logging
import re
import sqlite3
import threading
from pathlib import Path


SUPPORTED_SUFFIXES = {
    ".md",
    ".txt",
    ".rst",
    ".adoc",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
}
MAX_FILES = 500
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 10_000_000
CHUNK_CHARS = 1_200
MAX_QUERY_TERMS = 12
STOP_WORDS = {
    "about",
    "after",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "that",
    "the",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}

logger = logging.getLogger(__name__)


def _chunks(text: str):
    current: list[str] = []
    current_size = 0
    start_line = 1
    end_line = 1

    for line_number, line in enumerate(text.splitlines(), start=1):
        segments = [
            line[index : index + CHUNK_CHARS]
            for index in range(0, max(len(line), 1), CHUNK_CHARS)
        ]
        for segment in segments:
            if not segment.strip():
                continue
            separator_size = 1 if current else 0
            if current and current_size + separator_size + len(segment) > CHUNK_CHARS:
                yield "\n".join(current), start_line, end_line
                current = []
                current_size = 0
            if not current:
                start_line = line_number
            current.append(segment)
            current_size += len(segment) + (1 if len(current) > 1 else 0)
            end_line = line_number

    if current:
        yield "\n".join(current), start_line, end_line


def _query_expression(question: str) -> str:
    terms = []
    seen = set()
    for term in re.findall(r"\w{2,}", question.casefold()):
        if term not in STOP_WORDS and term not in seen:
            terms.append(term)
            seen.add(term)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in terms)


class DocumentIndex:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Workload documents directory not found: {self.root}")

        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                "path UNINDEXED, start_line UNINDEXED, end_line UNINDEXED, content)"
            )
        except sqlite3.OperationalError as exc:
            self._connection.close()
            raise RuntimeError("SQLite FTS5 support is required for document retrieval") from exc

        self.file_count = 0
        self.chunk_count = 0
        self._indexed_bytes = 0
        self._build()

    def _build(self):
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.suffix.casefold() in SUPPORTED_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(self.root).parts)
        )
        if len(paths) > MAX_FILES:
            logger.warning(
                "Workload index limited to %d files; found %d under %s",
                MAX_FILES,
                len(paths),
                self.root,
            )

        for path in paths[:MAX_FILES]:
            try:
                resolved = path.resolve(strict=True)
                relative_path = resolved.relative_to(self.root).as_posix()
                if not resolved.is_file():
                    continue
                size = resolved.stat().st_size
                if size > MAX_FILE_BYTES:
                    logger.warning("Skipping workload document over size limit: %s", relative_path)
                    continue
                if self._indexed_bytes + size > MAX_TOTAL_BYTES:
                    logger.warning(
                        "Workload index reached its %d-byte limit; remaining files are skipped",
                        MAX_TOTAL_BYTES,
                    )
                    break
                text = resolved.read_text(encoding="utf-8")
                if "\x00" in text:
                    logger.warning("Skipping non-text workload document: %s", relative_path)
                    continue
            except (OSError, UnicodeError, ValueError) as exc:
                logger.warning("Skipping unreadable workload document %s: %s", path, exc)
                continue

            rows = [
                (relative_path, start, end, content)
                for content, start, end in _chunks(text)
            ]
            if rows:
                self._connection.executemany(
                    "INSERT INTO chunks(path, start_line, end_line, content) VALUES (?, ?, ?, ?)",
                    rows,
                )
                self.file_count += 1
                self.chunk_count += len(rows)
                self._indexed_bytes += size

        self._connection.commit()
        logger.info(
            "Indexed %d chunks from %d workload documents in %s",
            self.chunk_count,
            self.file_count,
            self.root,
        )

    def search(self, question: str, limit: int = 3) -> list[dict[str, str | int]]:
        expression = _query_expression(question)
        if not expression:
            return []

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT path, start_line, end_line, content
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY bm25(chunks)
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        return [
            {
                "path": path,
                "start_line": int(start_line),
                "end_line": int(end_line),
                "content": content,
            }
            for path, start_line, end_line, content in rows
        ]

    def close(self):
        with self._lock:
            self._connection.close()
