"""S3-совместимое хранилище объектов.

Скачивание идёт потоком: предел размера обрывает чтение, а sha256 считается по
пути. Дочитывать гигабайт, чтобы затем сказать «слишком большой», значит уже
заплатить за него трафиком и диском, а второй проход ради суммы удвоил бы
ввод-вывод на каждом документе.

ETag за контрольную сумму не принимается: у многочастевой загрузки это хеш
хешей с суффиксом «-N», сверять с ним нечего.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import aioboto3
from aiobotocore.config import AioConfig
from botocore.exceptions import BotoCoreError, ClientError

from document_worker.application.errors import (
    ChecksumMismatchError,
    DocumentTooLargeError,
    SourceObjectNotFoundError,
)
from document_worker.application.ports.object_storage import ObjectStatDTO
from document_worker.domain.value_objects.storage import Checksum, ChecksumAlgorithm
from document_worker.infrastructure.storage.errors import translate_storage_error

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from types_aiobotocore_s3.client import S3Client

    from document_worker.domain.value_objects.storage import ObjectRef

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class S3Config:
    """Параметры подключения к хранилищу."""

    endpoint_url: str
    region: str
    access_key: str
    secret_key: str
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    max_attempts: int = 3


class S3ObjectStorage:
    """Доступ к S3-совместимому хранилищу поверх aioboto3."""

    def __init__(self, config: S3Config) -> None:
        """Готовит сессию; клиент открывается при входе в контекст."""
        self._config = config
        self._session = aioboto3.Session()
        self._exit_stack = contextlib.AsyncExitStack()
        self._client: S3Client | None = None

    async def __aenter__(self) -> Self:
        """Открывает клиента хранилища."""
        config = self._config
        self._client = await self._exit_stack.enter_async_context(
            self._session.client(
                "s3",
                endpoint_url=config.endpoint_url,
                region_name=config.region,
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                config=AioConfig(
                    connect_timeout=config.connect_timeout_s,
                    read_timeout=config.read_timeout_s,
                    retries={"max_attempts": config.max_attempts, "mode": "standard"},
                    signature_version="s3v4",
                ),
            )
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Закрывает клиента."""
        await self._exit_stack.aclose()
        self._client = None

    async def stat(self, ref: ObjectRef) -> ObjectStatDTO:
        """Метаданные объекта."""
        client = self._require_client()
        try:
            head = await client.head_object(Bucket=ref.bucket, Key=ref.key)
        except (ClientError, BotoCoreError) as error:
            raise translate_storage_error(error) from error
        return ObjectStatDTO(
            size_bytes=head["ContentLength"],
            content_type=head.get("ContentType"),
            etag=head.get("ETag"),
        )

    async def exists(self, ref: ObjectRef) -> bool:
        """Есть ли объект в хранилище."""
        try:
            await self.stat(ref)
        except SourceObjectNotFoundError:
            return False
        return True

    async def download_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
        timeout_s: float,
        expected_checksum: Checksum | None = None,
    ) -> Checksum:
        """Скачивает объект, обрывая поток при превышении предела.

        Raises:
            DocumentTooLargeError: Объект больше допустимого предела.
            ChecksumMismatchError: Скачанное не совпало с заявленной суммой.
        """
        try:
            async with asyncio.timeout(timeout_s):
                digest = await self._stream_to(ref, destination, max_bytes=max_bytes)
        except (ClientError, BotoCoreError) as error:
            _remove(destination)
            raise translate_storage_error(error) from error
        except BaseException:
            # Недокачанный файл нельзя оставлять: следующий шаг примет его
            # за целый документ.
            _remove(destination)
            raise

        checksum = Checksum(ChecksumAlgorithm.SHA256, digest)
        if expected_checksum is not None and not checksum.matches(expected_checksum):
            _remove(destination)
            raise ChecksumMismatchError(
                "контрольная сумма скачанного файла не совпала",
                context={
                    "expected": expected_checksum.value,
                    "actual": checksum.value,
                    "object_key": ref.key,
                },
            )
        return checksum

    async def _stream_to(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        max_bytes: int,
    ) -> str:
        """Пишет объект в файл, возвращая sha256 прочитанных байт."""
        client = self._require_client()
        response = await client.get_object(Bucket=ref.bucket, Key=ref.key)
        digest = hashlib.sha256()
        written = 0
        # Поток закрывается явно: оборванное на пределе соединение иначе
        # достаётся сборщику мусора вместе с предупреждением.
        body = response["Body"]
        async with body:
            with destination.open("wb") as target:
                async for chunk in body.iter_chunks(CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        raise DocumentTooLargeError(
                            "объект в хранилище больше допустимого предела",
                            context={"limit_bytes": max_bytes, "object_key": ref.key},
                        )
                    digest.update(chunk)
                    target.write(chunk)
        return digest.hexdigest()

    def _require_client(self) -> S3Client:
        if self._client is None:
            msg = "хранилище используется вне своего контекста"
            raise RuntimeError(msg)
        return self._client


def _remove(destination: Path) -> None:
    destination.unlink(missing_ok=True)
