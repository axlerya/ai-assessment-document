"""Состояние индексации документа в конкретной версии эмбеддингов.

Завершение идемпотентно намеренно. Протухший лиз допускает кратковременную
работу двух воркеров над одним документом: страницы и эмбеддинги не
задвоятся — их держат уникальные ограничения, — но до завершения дойдут оба.
Если бы второй вызов падал, корректно проиндексированный документ помечался бы
отказом. Поэтому повтор того же исхода — no-op, а противоречащий исход —
запрещённый переход.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC
from typing import TYPE_CHECKING, Self

from ai_worker.domain.errors import InvalidStatusTransition, InvariantViolation
from ai_worker.domain.value_objects.enums import IndexStatus
from ai_worker.domain.value_objects.identifiers import IndexId

if TYPE_CHECKING:
    from datetime import datetime

    from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
    from ai_worker.domain.value_objects.identifiers import (
        CorrelationId,
        DocumentId,
        EventId,
    )
    from ai_worker.domain.value_objects.source_snapshot import SourceSnapshot


def _require_utc(moment: datetime, *, field: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != UTC.utcoffset(None):
        raise InvariantViolation(
            f"{field} обязан быть моментом в UTC с указанием зоны",
            context={field: moment.isoformat()},
        )


@dataclass(frozen=True, slots=True)
class DocumentIndex:
    """Прогон индексации документа: статус, счётчики и происхождение корпуса."""

    id: IndexId
    document_id: DocumentId
    embedding: EmbeddingIdentity
    source: SourceSnapshot
    # Сообщение, с которого началась индексация: по нему разбирается, какая
    # доставка породила этот прогон.
    source_event_id: EventId
    status: IndexStatus
    correlation_id: CorrelationId | None = None
    chunks_total: int | None = None
    chunks_embedded: int = 0
    chunks_failed: int = 0
    failure_code: str | None = None
    failure_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def pending(
        cls,
        *,
        document_id: DocumentId,
        embedding: EmbeddingIdentity,
        source: SourceSnapshot,
        source_event_id: EventId,
        correlation_id: CorrelationId | None = None,
    ) -> Self:
        """Заводит прогон индексации с детерминированным ключом."""
        return cls(
            id=IndexId.deterministic(
                document_id=document_id, embedding_version=embedding.version
            ),
            document_id=document_id,
            embedding=embedding,
            source=source,
            source_event_id=source_event_id,
            status=IndexStatus.PENDING,
            correlation_id=correlation_id,
        )

    @property
    def is_terminal(self) -> bool:
        """Завершён ли прогон."""
        return self.status in IndexStatus.terminal()

    def start(self, *, at: datetime) -> Self:
        """Переводит прогон в работу.

        Raises:
            InvalidStatusTransition: Прогон уже начат или завершён.
        """
        _require_utc(at, field="started_at")
        if self.status is not IndexStatus.PENDING:
            self._refuse(IndexStatus.INDEXING)
        return replace(self, status=IndexStatus.INDEXING, started_at=at)

    def complete(
        self,
        *,
        chunks_total: int,
        chunks_embedded: int,
        chunks_failed: int,
        at: datetime,
    ) -> Self:
        """Завершает прогон успехом.

        Повтор того же завершения — no-op: второй воркер, дошедший до конца
        после протухшего лиза, не имеет права ни упасть, ни переписать чужой
        результат.

        Raises:
            InvalidStatusTransition: Прогон не в работе и не завершён успешно.
            InvariantViolation: Счётчики не сходятся либо ни один чанк не
                проиндексирован.
        """
        _require_utc(at, field="finished_at")
        if self.status is IndexStatus.INDEXED:
            return self
        if self.status is not IndexStatus.INDEXING:
            self._refuse(IndexStatus.INDEXED)
        self._validate_counters(
            chunks_total=chunks_total,
            chunks_embedded=chunks_embedded,
            chunks_failed=chunks_failed,
        )
        self._validate_finish(at)
        return replace(
            self,
            status=IndexStatus.INDEXED,
            chunks_total=chunks_total,
            chunks_embedded=chunks_embedded,
            chunks_failed=chunks_failed,
            finished_at=at,
        )

    def fail(
        self,
        *,
        code: str,
        message: str,
        at: datetime,
        chunks_total: int | None = None,
        chunks_failed: int = 0,
    ) -> Self:
        """Завершает прогон отказом.

        Счётчики записываются и здесь: строка отказа без них не объясняет, что
        именно не вышло, а это первое, что спрашивают при разборе. Проверка на
        сходимость к ним не применяется — успешных чанков в отказе нет по
        определению.

        Raises:
            InvalidStatusTransition: Прогон уже завершён успешно — это не
                повтор, а противоречие.
            InvariantViolation: Код отказа пуст либо завершение раньше старта.
        """
        _require_utc(at, field="finished_at")
        if self.status is IndexStatus.FAILED:
            return self
        if self.status is not IndexStatus.INDEXING:
            self._refuse(IndexStatus.FAILED)
        if not code.strip():
            raise InvariantViolation(
                "отказ без кода нельзя разобрать",
                context={"document_id": str(self.document_id)},
            )
        self._validate_finish(at)
        return replace(
            self,
            status=IndexStatus.FAILED,
            chunks_total=chunks_total,
            chunks_failed=chunks_failed,
            failure_code=code,
            failure_message=message,
            finished_at=at,
        )

    def _refuse(self, target: IndexStatus) -> None:
        raise InvalidStatusTransition(
            f"переход {self.status.value} → {target.value} запрещён",
            context={"from": self.status.value, "to": target.value},
        )

    def _validate_counters(
        self,
        *,
        chunks_total: int,
        chunks_embedded: int,
        chunks_failed: int,
    ) -> None:
        if min(chunks_total, chunks_embedded, chunks_failed) < 0:
            raise InvariantViolation(
                "счётчик чанков отрицателен",
                context={
                    "total": chunks_total,
                    "embedded": chunks_embedded,
                    "failed": chunks_failed,
                },
            )
        if chunks_embedded + chunks_failed != chunks_total:
            raise InvariantViolation(
                "счётчики чанков не сходятся с их общим числом",
                context={
                    "total": chunks_total,
                    "counted": chunks_embedded + chunks_failed,
                },
            )
        if chunks_embedded == 0:
            # Иначе поиск нашёл бы по документу ровно ничего и молча вернул
            # пустой контекст, а документ считался бы готовым.
            raise InvariantViolation(
                "документ без единого построенного эмбеддинга не проиндексирован",
                context={"total": chunks_total, "failed": chunks_failed},
            )

    def _validate_finish(self, at: datetime) -> None:
        if self.started_at is not None and at < self.started_at:
            raise InvariantViolation(
                "завершение раньше старта",
                context={
                    "started_at": self.started_at.isoformat(),
                    "finished_at": at.isoformat(),
                },
            )
