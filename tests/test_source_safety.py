import io
import unittest
import zipfile

from rag_platform.services.source_safety import UnsafeSourceError, inspect_zip, validate_document


def make_zip(entries: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


class SourceSafetyTests(unittest.TestCase):
    def test_plain_text_is_allowed(self) -> None:
        self.assertEqual(validate_document("notes.txt", b"safe", "text/plain"), ".txt")

    def test_executable_is_rejected(self) -> None:
        with self.assertRaises(UnsafeSourceError):
            validate_document("payload.exe", b"inert")

    def test_mime_mismatch_is_rejected(self) -> None:
        with self.assertRaises(UnsafeSourceError):
            validate_document("report.pdf", b"not-a-pdf", "text/plain")

    def test_control_character_in_filename_is_rejected(self) -> None:
        with self.assertRaises(UnsafeSourceError):
            validate_document("unsafe\x00.txt", b"inert")

    def test_archive_path_traversal_is_rejected(self) -> None:
        content = make_zip({"../escape.txt": b"inert"})
        with self.assertRaises(UnsafeSourceError):
            inspect_zip(content)

    def test_nested_allowed_text_is_returned(self) -> None:
        nested = make_zip({"folder/notes.md": b"safe text"})
        outer = make_zip({"nested.zip": nested})
        members = inspect_zip(outer)
        self.assertEqual(
            [(member.path, member.content) for member in members],
            [("folder/notes.md", b"safe text")],
        )

    def test_archive_depth_over_three_is_rejected(self) -> None:
        level_four = make_zip({"notes.txt": b"safe text"})
        level_three = make_zip({"level-four.zip": level_four})
        level_two = make_zip({"level-three.zip": level_three})
        level_one = make_zip({"level-two.zip": level_two})
        with self.assertRaises(UnsafeSourceError):
            inspect_zip(level_one)


if __name__ == "__main__":
    unittest.main()
