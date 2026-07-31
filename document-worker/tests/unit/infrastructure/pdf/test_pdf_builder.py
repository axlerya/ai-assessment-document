"""Генератор тестовых PDF обязан быть детерминированным.

Фикстура, меняющаяся от запуска к запуску, превращает падение теста в лотерею:
непонятно, сломался код или сгенерировалось другое.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pikepdf
import pytest

from tests.fakes import pdf_builder

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.unit

BUILDERS: tuple[Callable[[Path], Path], ...] = (
    pdf_builder.make_text_pdf,
    pdf_builder.make_two_column_pdf,
    pdf_builder.make_glued_text_pdf,
    pdf_builder.make_broken_tounicode_pdf,
    pdf_builder.make_empty_pdf,
    pdf_builder.make_corrupted_pdf,
    pdf_builder.make_truncated_pdf,
    pdf_builder.make_scan_pdf,
    pdf_builder.make_ccitt_g4_scan_pdf,
)


@pytest.mark.parametrize("build", BUILDERS, ids=lambda build: build.__name__)
def test_generator_is_deterministic(
    build: Callable[[Path], Path],
    tmp_path: Path,
) -> None:
    first = build(tmp_path / "first.pdf").read_bytes()
    second = build(tmp_path / "second.pdf").read_bytes()

    assert first == second


def test_text_pdf_has_the_requested_number_of_pages(tmp_path: Path) -> None:
    path = pdf_builder.make_text_pdf(tmp_path / "doc.pdf", pages=3)

    with pikepdf.open(path) as pdf:
        assert len(pdf.pages) == 3


def test_empty_pdf_has_no_pages(tmp_path: Path) -> None:
    path = pdf_builder.make_empty_pdf(tmp_path / "doc.pdf")

    with pikepdf.open(path) as pdf:
        assert len(pdf.pages) == 0


def test_corrupted_pdf_needs_recovery_to_open(tmp_path: Path) -> None:
    # Сломана только таблица ссылок: qpdf пересканирует объекты и соберёт файл.
    path = pdf_builder.make_corrupted_pdf(tmp_path / "doc.pdf")

    with pikepdf.open(path) as pdf:
        assert len(pdf.pages) == 1


def test_truncated_pdf_cannot_be_opened(tmp_path: Path) -> None:
    path = pdf_builder.make_truncated_pdf(tmp_path / "doc.pdf")

    with pytest.raises(pikepdf.PdfError):
        pikepdf.open(path).close()


@pytest.mark.parametrize(
    ("build", "password"),
    [
        (pdf_builder.make_encrypted_pdf, "secret"),
        (pdf_builder.make_owner_encrypted_pdf, ""),
    ],
    ids=["user_password", "owner_password"],
)
def test_encrypted_generators_repeat_their_content(
    build: Callable[[Path], Path],
    password: str,
    tmp_path: Path,
) -> None:
    # Байты у шифрованного файла различаются от запуска к запуску: ключ
    # выводится из случайного идентификатора. Повторяется содержимое.
    first = build(tmp_path / "first.pdf")
    second = build(tmp_path / "second.pdf")

    with (
        pikepdf.open(first, password=password) as one,
        pikepdf.open(second, password=password) as two,
    ):
        assert len(one.pages) == len(two.pages) == 1


def test_encrypted_pdf_needs_the_user_password(tmp_path: Path) -> None:
    path = pdf_builder.make_encrypted_pdf(tmp_path / "doc.pdf")

    with pytest.raises(pikepdf.PasswordError):
        pikepdf.open(path).close()


def test_owner_encrypted_pdf_opens_without_a_password(tmp_path: Path) -> None:
    path = pdf_builder.make_owner_encrypted_pdf(tmp_path / "doc.pdf")

    with pikepdf.open(path) as pdf:
        assert pdf.is_encrypted


def test_scan_pdf_carries_an_image_and_no_text(tmp_path: Path) -> None:
    path = pdf_builder.make_scan_pdf(tmp_path / "doc.pdf")

    with pikepdf.open(path) as pdf:
        resources = pdf.pages[0].Resources
        assert "/XObject" in resources
        assert "/Font" not in resources


def test_ccitt_scan_declares_its_codec(tmp_path: Path) -> None:
    path = pdf_builder.make_ccitt_g4_scan_pdf(tmp_path / "doc.pdf")

    with pikepdf.open(path) as pdf:
        image = pdf.pages[0].Resources.XObject.Im1
        assert image.Filter == pikepdf.Name.CCITTFaxDecode


def test_non_pdf_file_does_not_start_with_the_signature(tmp_path: Path) -> None:
    path = pdf_builder.make_non_pdf_file(tmp_path / "doc.pdf")

    assert not path.read_bytes().startswith(b"%PDF-")
