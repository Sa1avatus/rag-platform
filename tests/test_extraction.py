import io
import json
import unittest
import zipfile

from rag_platform.services.extraction import extract


class ExtractionTests(unittest.TestCase):
    def test_html_omits_scripts_and_normalizes_text(self) -> None:
        result = extract(
            "page.html",
            b"<h1>Title</h1><script>secret()</script><p>Body</p>",
        )
        self.assertEqual(result[0].content, "Title Body")

    def test_json_is_canonicalized(self) -> None:
        result = extract("record.json", b'{"z": 1, "a": "text"}')
        self.assertEqual(json.loads(result[0].content), {"a": "text", "z": 1})
        self.assertLess(result[0].content.index('"a"'), result[0].content.index('"z"'))

    def test_email_extracts_headers_and_plain_body(self) -> None:
        content = (
            b"Subject: Example\r\n"
            b"From: sender@example.test\r\n"
            b"To: receiver@example.test\r\n\r\n"
            b"Message body"
        )
        result = extract("message.eml", content)
        self.assertIn("Subject: Example", result[0].content)
        self.assertIn("Message body", result[0].content)

    def test_zip_returns_each_safe_member(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("one.txt", "First")
            archive.writestr("two.md", "Second")
        result = extract("documents.zip", output.getvalue())
        self.assertEqual(
            [(item.filename, item.content) for item in result],
            [("one.txt", "First"), ("two.md", "Second")],
        )


if __name__ == "__main__":
    unittest.main()
