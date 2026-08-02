"""Шрифт корпуса: один, закреплённый суммой, скачиваемый.

Корпус кириллический — сервис обрабатывает русские документы, и распознавание
у него восточнославянское. Ни Pillow, ни rapidocr шрифта с кириллицей не
несут, а системный брать нельзя: на разных машинах он разный, растр
получился бы разным, и `corpus_hash` перестал бы сходиться — то есть защита
корпуса от упрощения отключилась бы молча. Запасного варианта нет намеренно:
подстановка «похожего» шрифта и есть та самая молчаливая порча.

В репозиторий шрифт не кладётся: он качается и проверяется тем же приёмом, что
модели распознавания.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

FONT_DIR_ENV: Final[str] = "EVAL__FONT_DIR"
DEFAULT_FONT_DIR: Final[Path] = Path(".fonts")

ARCHIVE_URL: Final[str] = (
    "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.zip"
)
ARCHIVE_SHA256: Final[str] = (
    "7576310b219e04159d35ff61dd4a4ec4cdba4f35c00e002a136f00e96a908b0a"
)
FONT_VERSION: Final[str] = "dejavu-2.37"

_DOWNLOAD_TIMEOUT_S: Final[float] = 180.0
_CHUNK_BYTES: Final[int] = 1024 * 1024


class CorpusFontsUnavailableError(RuntimeError):
    """Шрифта корпуса нет на месте или он подменён."""


@dataclass(frozen=True, slots=True)
class FontFile:
    """Один шрифт: роль, имя файла, путь внутри архива и сумма."""

    name: str
    file_name: str
    member: str
    sha256: str


REQUIRED_FONTS: Final[tuple[FontFile, ...]] = (
    FontFile(
        name="sans",
        file_name="DejaVuSans.ttf",
        member="dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
        sha256="7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954",
    ),
    FontFile(
        name="sans-bold",
        file_name="DejaVuSans-Bold.ttf",
        member="dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf",
        sha256="e6476c1b80502924294eed40894c5b18e06c181444ca953e5334262df9c27724",
    ),
    FontFile(
        name="serif",
        file_name="DejaVuSerif.ttf",
        member="dejavu-fonts-ttf-2.37/ttf/DejaVuSerif.ttf",
        sha256="42d1edeb7952f31b1f96d767ed7030b08a39e0c372b0071641518864e2bffb51",
    ),
)


def font_dir_from_env() -> Path:
    """Каталог шрифтов корпуса."""
    return Path(os.environ.get(FONT_DIR_ENV) or DEFAULT_FONT_DIR)


def resolve(font_dir: Path) -> Mapping[str, Path]:
    """Пути ко всем объявленным шрифтам."""
    return {font.name: font_dir / font.file_name for font in REQUIRED_FONTS}


def missing_fonts(font_dir: Path) -> tuple[FontFile, ...]:
    """Шрифты, которых нет на месте или чья сумма не сошлась."""
    return tuple(
        font
        for font in REQUIRED_FONTS
        if _checksum_of(font_dir / font.file_name) != font.sha256
    )


def verify(font_dir: Path) -> None:
    """Требует, чтобы все шрифты были на месте и не подменены.

    Raises:
        CorpusFontsUnavailableError: Файла нет либо его сумма не совпала.
    """
    for font in REQUIRED_FONTS:
        path = font_dir / font.file_name
        if not path.is_file():
            raise CorpusFontsUnavailableError(
                f"шрифта корпуса нет на месте: {path}",
            )
        actual = _checksum_of(path)
        if actual != font.sha256:
            raise CorpusFontsUnavailableError(
                f"сумма шрифта {font.file_name} не совпала:"
                f" ожидалась {font.sha256}, получена {actual}",
            )


def checksum() -> str:
    """Сумма набора шрифтов — часть отпечатка корпуса."""
    digest = hashlib.sha256()
    for font in REQUIRED_FONTS:
        digest.update(f"{font.name}:{font.sha256}\n".encode())
    return digest.hexdigest()


def download_missing(font_dir: Path) -> tuple[str, ...]:
    """Достаёт недостающие шрифты из архива и возвращает их роли.

    Raises:
        CorpusFontsUnavailableError: Сумма архива или шрифта не совпала.
    """
    absent = missing_fonts(font_dir)
    if not absent:
        return ()
    font_dir.mkdir(parents=True, exist_ok=True)
    payload = _fetch()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise CorpusFontsUnavailableError(
            f"сумма архива шрифтов не совпала:"
            f" ожидалась {ARCHIVE_SHA256}, получена {actual}",
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for font in absent:
            _extract(archive, font, font_dir / font.file_name)
    return tuple(font.name for font in absent)


def _fetch() -> bytes:
    with urllib.request.urlopen(ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        payload: bytes = response.read()
    return payload


def _extract(archive: zipfile.ZipFile, font: FontFile, destination: Path) -> None:
    payload = archive.read(font.member)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != font.sha256:
        raise CorpusFontsUnavailableError(
            f"сумма шрифта {font.file_name} в архиве не совпала:"
            f" ожидалась {font.sha256}, получена {actual}",
        )
    destination.write_bytes(payload)


def _checksum_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
