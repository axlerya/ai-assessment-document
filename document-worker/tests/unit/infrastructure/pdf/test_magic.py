"""Определение типа файла по сигнатуре.

Расширение и заявленный MIME приходят снаружи и врут: тип берётся из первых
байт самого файла.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_worker.infrastructure.pdf.magic import detect_mime_type, is_pdf
from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_generated_pdf_is_recognized(tmp_path: Path) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf")

    assert is_pdf(path)
    assert detect_mime_type(path) == "application/pdf"


def test_file_with_pdf_extension_but_other_content_is_not_a_pdf(
    tmp_path: Path,
) -> None:
    path = pdf_builder.make_non_pdf_file(tmp_path / "doc.pdf")

    assert not is_pdf(path)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"%PDF-1.7\n...", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"PK\x03\x04", "application/zip"),
        (b"II*\x00", "image/tiff"),
        (b"MM\x00*", "image/tiff"),
        (b"something else entirely", None),
    ],
)
def test_known_signatures_are_recognized(
    payload: bytes,
    expected: str | None,
    tmp_path: Path,
) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(payload)

    assert detect_mime_type(path) == expected


def test_empty_file_has_no_type(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert detect_mime_type(path) is None
    assert not is_pdf(path)


def test_missing_file_has_no_type(tmp_path: Path) -> None:
    assert detect_mime_type(tmp_path / "nowhere.bin") is None


def test_signature_is_read_without_loading_the_whole_file(tmp_path: Path) -> None:
    # Сигнатура лежит в первых байтах, и читать ради неё гигабайт незачем.
    path = tmp_path / "big.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(5 * 1024 * 1024))

    assert is_pdf(path)
