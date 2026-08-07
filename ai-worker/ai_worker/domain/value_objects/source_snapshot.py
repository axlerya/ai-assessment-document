"""Что именно произвёл document-worker по этому документу.

Три величины всегда ходят вместе: версия обработки, версия чанкования и исход.
Версия чанкования выбирает корпус, версия обработки объясняет его качество, а
исход решает, есть ли смысл индексировать вообще. Порознь ни одна из них не
отвечает на вопрос «что мы индексируем», поэтому это одно значение.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.domain.value_objects.enums import SourceStatus

if TYPE_CHECKING:
    from ai_worker.domain.value_objects.versioning import (
        ChunkingVersion,
        PipelineVersion,
    )


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Состояние документа после обработки соседним сервисом."""

    pipeline_version: PipelineVersion
    chunking_version: ChunkingVersion
    status: SourceStatus

    @property
    def is_complete(self) -> bool:
        """Прочитан ли документ целиком.

        Частично прочитанный индексируется наравне с полным: выбросить его
        значило бы потерять из корпуса любой скан с одной нечитаемой строкой.
        Отличие в том, что его чанки чаще несут пометки ненадёжности.
        """
        return self.status is SourceStatus.PROCESSED
