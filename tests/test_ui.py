import unittest

from ui import build_messages, safe_history


class ChatHistoryTests(unittest.TestCase):
    def test_gradio_message_history_is_preserved(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        messages = build_messages(history, "How are you?")

        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant running locally.",
                },
                *history,
                {"role": "user", "content": "How are you?"},
            ],
        )

    def test_malformed_history_items_are_ignored(self):
        self.assertEqual(
            safe_history([{"role": "system", "content": "ignored"}, None]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
