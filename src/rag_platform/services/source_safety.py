import io
import mimetypes
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

DOCUMENT_MAX_BYTES = 25 * 1024 * 1024
ARCHIVE_MAX_BYTES = 25 * 1024 * 1024
ARCHIVE_EXPANDED_MAX_BYTES = 100 * 1024 * 1024
ARCHIVE_MAX_DEPTH = 3

ALLOWED_EXTENSIONS = frozenset(
    {
        ".csv",
        ".docx",
        ".eml",
        ".html",
        ".htm",
        ".json",
        ".md",
        ".pdf",
        ".py",
        ".rst",
        ".text",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
        ".zip",
    }
)
EXECUTABLE_EXTENSIONS = frozenset(
    {
        ".bat",
        ".cmd",
        ".com",
        ".dll",
        ".exe",
        ".jar",
        ".msi",
        ".ps1",
        ".scr",
        ".sh",
    }
)


class UnsafeSourceError(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    content: bytes


def validate_document(filename: str, content: bytes, mime_type: str | None = None) -> str:
    if not filename or any(ord(character) < 32 for character in filename):
        raise UnsafeSourceError("filename is empty or contains control characters")
    if len(content) > DOCUMENT_MAX_BYTES:
        raise UnsafeSourceError("document is empty or exceeds the 25 MiB limit")
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if suffix in EXECUTABLE_EXTENSIONS or suffix not in ALLOWED_EXTENSIONS:
        raise UnsafeSourceError("file extension is not allowed")
    expected = mimetypes.guess_type(filename)[0]
    if mime_type and expected and not _mime_compatible(suffix, expected, mime_type):
        raise UnsafeSourceError("MIME type does not match the file extension")
    return suffix


def inspect_zip(
    content: bytes,
    depth: int = 1,
    expanded_budget: list[int] | None = None,
) -> list[ArchiveMember]:
    if depth > ARCHIVE_MAX_DEPTH:
        raise UnsafeSourceError("archive nesting exceeds the maximum depth")
    if len(content) > ARCHIVE_MAX_BYTES:
        raise UnsafeSourceError("archive exceeds the 25 MiB limit")
    members: list[ArchiveMember] = []
    if expanded_budget is None:
        expanded_budget = [0]
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsafeSourceError("invalid ZIP archive") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            _validate_member_path(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise UnsafeSourceError("symbolic links are not allowed in archives")
            expanded_budget[0] += info.file_size
            if expanded_budget[0] > ARCHIVE_EXPANDED_MAX_BYTES:
                raise UnsafeSourceError("expanded archive exceeds the 100 MiB limit")
            if info.compress_size == 0 and info.file_size > 0:
                raise UnsafeSourceError("suspicious archive compression ratio")
            if info.compress_size and info.file_size / info.compress_size > 200:
                raise UnsafeSourceError("suspicious archive compression ratio")
            payload = archive.read(info)
            suffix = validate_document(info.filename, payload)
            if suffix == ".zip":
                members.extend(inspect_zip(payload, depth + 1, expanded_budget))
            else:
                members.append(ArchiveMember(info.filename, payload))
    return members


def _validate_member_path(filename: str) -> None:
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or posixpath.normpath(normalized).startswith("../"):
        raise UnsafeSourceError("archive member escapes its extraction root")
    if not path.name:
        raise UnsafeSourceError("archive member has an invalid path")


def _mime_compatible(suffix: str, expected: str, supplied: str) -> bool:
    supplied = supplied.split(";", 1)[0].strip().lower()
    if suffix in {".md", ".rst", ".text", ".txt"}:
        return supplied.startswith("text/")
    if suffix in {".yaml", ".yml"}:
        return supplied in {"application/x-yaml", "application/yaml", "text/yaml", "text/plain"}
    return supplied == expected.lower()
