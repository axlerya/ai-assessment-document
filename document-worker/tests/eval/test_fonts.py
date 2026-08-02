"""Шрифт корпуса: закреплённая сумма и никакого системного запасного варианта.

Системный шрифт сделал бы растр машинно-зависимым, а вместе с ним и
`corpus_hash` — то есть отключил бы защиту корпуса от упрощения.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from typing import TYPE_CHECKING

import pytest

from eval import fonts
from eval.fonts import (
    REQUIRED_FONTS,
    CorpusFontsUnavailableError,
    FontFile,
    missing_fonts,
    resolve,
    verify,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

PAYLOAD = b"ttf-font-bytes"
MEMBER = "dejavu/ttf/Sample.ttf"
FILE_NAME = "Sample.ttf"
ARCHIVE_MEMBERS = {MEMBER: PAYLOAD, "dejavu/README": b"docs"}


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    # Отметки времени внутри архива фиксированы: иначе сумма архива менялась бы
    # от запуска к запуску и проверять было бы нечего.
    with zipfile.ZipFile(buffer, "w") as target:
        for name, payload in members.items():
            target.writestr(zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)), payload)
    return buffer.getvalue()


@pytest.fixture
def declared(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Подменяет реестр одним шрифтом и возвращает архив с ним."""
    archive = _archive(ARCHIVE_MEMBERS)
    monkeypatch.setattr(
        fonts,
        "REQUIRED_FONTS",
        (
            FontFile(
                name="sans",
                file_name=FILE_NAME,
                member=MEMBER,
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
        ),
    )
    monkeypatch.setattr(fonts, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    return archive


@pytest.fixture
def font_dir(declared: bytes, tmp_path: Path) -> Path:  # noqa: ARG001 — реестр подменяет фикстура
    """Каталог, в котором объявленный шрифт уже лежит."""
    (tmp_path / FILE_NAME).write_bytes(PAYLOAD)
    return tmp_path


def test_every_declared_font_has_a_member_and_a_hash() -> None:
    assert REQUIRED_FONTS
    for font in REQUIRED_FONTS:
        assert font.member
        assert len(font.sha256) == 64


def test_declared_fonts_come_from_one_pinned_archive() -> None:
    assert fonts.ARCHIVE_URL.startswith("https://")
    assert len(fonts.ARCHIVE_SHA256) == 64


def test_verify_passes_on_complete_directory(font_dir: Path) -> None:
    verify(font_dir)


def test_missing_font_is_refused(font_dir: Path) -> None:
    (font_dir / FILE_NAME).unlink()

    with pytest.raises(CorpusFontsUnavailableError):
        verify(font_dir)


def test_replaced_font_is_refused(font_dir: Path) -> None:
    # Подменённый шрифт рисует другие глифы и делает это молча.
    (font_dir / FILE_NAME).write_bytes(b"other-font")

    with pytest.raises(CorpusFontsUnavailableError):
        verify(font_dir)


def test_resolve_returns_path_of_every_font(font_dir: Path) -> None:
    assert resolve(font_dir) == {"sans": font_dir / FILE_NAME}


def test_missing_fonts_lists_what_has_to_be_downloaded(
    declared: bytes,  # noqa: ARG001 — реестр подменяет фикстура
    tmp_path: Path,
) -> None:
    assert [font.name for font in missing_fonts(tmp_path)] == ["sans"]


def test_download_extracts_and_verifies(
    declared: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fonts, "_fetch", lambda: declared)

    assert fonts.download_missing(tmp_path) == ("sans",)
    assert (tmp_path / FILE_NAME).read_bytes() == PAYLOAD


def test_download_skips_fonts_already_in_place(
    font_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _refuse() -> bytes:
        pytest.fail("архив скачан, хотя шрифт уже на месте")

    monkeypatch.setattr(fonts, "_fetch", _refuse)

    assert fonts.download_missing(font_dir) == ()


def test_download_of_a_replaced_archive_is_refused(
    declared: bytes,  # noqa: ARG001 — реестр подменяет фикстура
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fonts, "_fetch", lambda: _archive({MEMBER: b"tampered"}))

    with pytest.raises(CorpusFontsUnavailableError):
        fonts.download_missing(tmp_path)


def test_download_of_a_replaced_font_inside_the_archive_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Сумма архива сошлась, а файл внутри не тот: проверяются оба уровня.
    archive = _archive({MEMBER: b"tampered"})
    monkeypatch.setattr(
        fonts,
        "REQUIRED_FONTS",
        (
            FontFile(
                name="sans",
                file_name=FILE_NAME,
                member=MEMBER,
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
        ),
    )
    monkeypatch.setattr(fonts, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(fonts, "_fetch", lambda: archive)

    with pytest.raises(CorpusFontsUnavailableError):
        fonts.download_missing(tmp_path)
