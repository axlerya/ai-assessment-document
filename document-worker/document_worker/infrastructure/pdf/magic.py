"""Тип файла по сигнатуре.

Расширение и заявленный MIME приходят снаружи и врут. Отдельная библиотека для
таблицы из шести строк не нужна — она добавила бы зависимость и ничего больше.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

PDF_MIME_TYPE: Final[str] = "application/pdf"

# Порядок важен: более длинная сигнатура проверяется раньше короткой.
_SIGNATURES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"%PDF-", PDF_MIME_TYPE),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"PK\x03\x04", "application/zip"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)

_PREFIX_SIZE: Final[int] = 16


def detect_mime_type(path: Path) -> str | None:
    """Определяет тип по первым байтам файла.

    Возвращает None, если сигнатура неизвестна или файла нет: отсутствие типа
    это ответ, а не ошибка чтения.
    """
    prefix = _prefix_of(path)
    if not prefix:
        return None
    for signature, mime_type in _SIGNATURES:
        if prefix.startswith(signature):
            return mime_type
    return None


def is_pdf(path: Path) -> bool:
    """PDF ли это по содержимому, а не по имени."""
    return detect_mime_type(path) == PDF_MIME_TYPE


def _prefix_of(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            return source.read(_PREFIX_SIZE)
    except OSError:
        return b""
