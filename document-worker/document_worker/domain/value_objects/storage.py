"""Исходный файл документа: ссылка на объект, контрольная сумма, тип и размер."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final, NoReturn, Self

from document_worker.domain.constants import (
    MAX_BUCKET_NAME_LENGTH,
    MAX_FILE_SIZE_BYTES,
    MAX_OBJECT_KEY_LENGTH,
    MIN_BUCKET_NAME_LENGTH,
    SUPPORTED_MIME_TYPES,
)
from document_worker.domain.errors import (
    ChecksumMismatch,
    DocumentTooLarge,
    EmptyDocument,
    InvalidChecksum,
    InvalidFileSize,
    InvalidMimeType,
    InvalidObjectRef,
    UnsupportedDocumentFormat,
)

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_MIME_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")

_S3_SCHEME: Final[str] = "s3://"


class ChecksumAlgorithm(StrEnum):
    """Алгоритм контрольной суммы."""

    SHA256 = "sha256"


_HEX_LENGTH_BY_ALGORITHM: Final[dict[ChecksumAlgorithm, int]] = {
    ChecksumAlgorithm.SHA256: 64,
}


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Ссылка на объект в S3-совместимом хранилище."""

    bucket: str
    key: str

    def __post_init__(self) -> None:
        """Проверяет имя бакета и безопасность ключа."""
        self._validate_bucket()
        self._validate_key()

    def _validate_bucket(self) -> None:
        if not MIN_BUCKET_NAME_LENGTH <= len(self.bucket) <= MAX_BUCKET_NAME_LENGTH:
            self._reject("длина имени бакета", self.bucket)
        if _BUCKET_RE.match(self.bucket) is None:
            self._reject("имя бакета", self.bucket)
        if ".." in self.bucket:
            self._reject("две точки подряд в имени бакета", self.bucket)
        # Имя, похожее на IPv4, S3 отвергает: оно неотличимо от адреса эндпоинта.
        if _IPV4_RE.match(self.bucket) is not None:
            self._reject("имя бакета в форме IPv4", self.bucket)

    def _validate_key(self) -> None:
        if not 1 <= len(self.key) <= MAX_OBJECT_KEY_LENGTH:
            self._reject("длина ключа", self.key)
        if self.key.startswith("/"):
            self._reject("ведущий слэш в ключе", self.key)
        if "\\" in self.key:
            self._reject("обратный слэш в ключе", self.key)
        if any(segment == ".." for segment in self.key.split("/")):
            self._reject("выход за пределы префикса", self.key)
        if any(char < " " or char == "\x7f" for char in self.key):
            self._reject("управляющий символ в ключе", self.key)

    @staticmethod
    def _reject(reason: str, value: str) -> NoReturn:
        raise InvalidObjectRef(
            f"недопустимая ссылка на объект: {reason}",
            context={"reason": reason, "value": value},
        )

    @classmethod
    def parse_uri(cls, uri: str) -> Self:
        """Разбирает ссылку вида `s3://bucket/key`."""
        if not uri.startswith(_S3_SCHEME):
            cls._reject("схема ссылки", uri)
        bucket, separator, key = uri[len(_S3_SCHEME) :].partition("/")
        if not separator:
            cls._reject("в ссылке нет ключа", uri)
        return cls(bucket=bucket, key=key)

    def to_uri(self) -> str:
        """Собирает ссылку вида `s3://bucket/key`."""
        return f"{_S3_SCHEME}{self.bucket}/{self.key}"


@dataclass(frozen=True, slots=True)
class Checksum:
    """Контрольная сумма файла. Значение хранится в нижнем регистре."""

    algorithm: ChecksumAlgorithm
    value: str

    def __post_init__(self) -> None:
        """Нормализует значение и проверяет его длину."""
        normalized = self.value.strip().lower()
        expected_length = _HEX_LENGTH_BY_ALGORITHM[self.algorithm]
        if len(normalized) != expected_length or _HEX_RE.match(normalized) is None:
            raise InvalidChecksum(
                f"{self.algorithm.value} требует {expected_length} hex-символов",
                context={"algorithm": self.algorithm.value, "length": len(normalized)},
            )
        object.__setattr__(self, "value", normalized)

    @classmethod
    def sha256_of(cls, data: bytes) -> Self:
        """Считает sha256 от переданных байт."""
        return cls(ChecksumAlgorithm.SHA256, hashlib.sha256(data).hexdigest())

    def matches(self, other: Checksum) -> bool:
        """Совпадают ли суммы. Разные алгоритмы сравнению не подлежат."""
        return (self.algorithm, self.value) == (other.algorithm, other.value)


@dataclass(frozen=True, slots=True)
class MimeType:
    """MIME-тип без параметров, в нижнем регистре."""

    value: str

    PDF: ClassVar[str] = "application/pdf"

    def __post_init__(self) -> None:
        """Отбрасывает параметры после `;` и проверяет форму типа."""
        normalized = self.value.split(";", maxsplit=1)[0].strip().lower()
        if _MIME_RE.match(normalized) is None:
            raise InvalidMimeType(
                f"{self.value!r} не является MIME-типом",
                context={"value": self.value},
            )
        object.__setattr__(self, "value", normalized)

    @property
    def is_pdf(self) -> bool:
        """PDF ли это."""
        return self.value == MimeType.PDF

    def is_supported(self) -> bool:
        """Входит ли тип в список поддерживаемых."""
        return self.value in SUPPORTED_MIME_TYPES

    def ensure_supported(self) -> None:
        """Проверяет поддержку типа.

        Raises:
            UnsupportedDocumentFormat: Тип вне списка поддерживаемых.
        """
        if not self.is_supported():
            raise UnsupportedDocumentFormat(self.value, supported=SUPPORTED_MIME_TYPES)


@dataclass(frozen=True, slots=True, order=True)
class FileSize:
    """Размер файла в байтах. Верхний предел задаётся вызывающим."""

    value: int

    def __post_init__(self) -> None:
        """Отвергает отрицательный размер."""
        if self.value < 0:
            raise InvalidFileSize(
                "размер файла отрицателен",
                context={"value": self.value},
            )

    def __int__(self) -> int:
        """Возвращает размер в байтах."""
        return self.value

    def ensure_within(self, limit_bytes: int) -> None:
        """Проверяет размер против предела.

        Raises:
            DocumentTooLarge: Размер больше предела.
        """
        if self.value > limit_bytes:
            raise DocumentTooLarge(actual_bytes=self.value, limit_bytes=limit_bytes)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Исходный файл документа в хранилище."""

    ref: ObjectRef
    mime_type: MimeType
    size: FileSize
    checksum: Checksum | None = None

    def ensure_acceptable(self) -> None:
        """Проверяет, что файл вообще пригоден к обработке.

        Raises:
            EmptyDocument: Нулевой размер.
            UnsupportedDocumentFormat: Формат вне списка поддерживаемых.
            DocumentTooLarge: Размер выше предела.
        """
        if self.size.value == 0:
            raise EmptyDocument(
                "документ нулевого размера",
                context={"object_key": self.ref.key},
            )
        self.mime_type.ensure_supported()
        self.size.ensure_within(MAX_FILE_SIZE_BYTES)

    def verify(self, actual: Checksum) -> None:
        """Сверяет заявленную сумму с фактической, если заявленная известна.

        Raises:
            ChecksumMismatch: Суммы не совпали.
        """
        if self.checksum is None:
            return
        if not self.checksum.matches(actual):
            raise ChecksumMismatch(expected=self.checksum.value, actual=actual.value)
