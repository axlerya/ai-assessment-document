"""Где именно считается модель.

Провайдер отделён от прогона намеренно: пачки, префиксы, нормировка и обрезка
проверяются без модели, а всё, что требует двух гигабайт весов, остаётся за
этим узким швом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ai_worker.infrastructure.embedding import encoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ai_worker.infrastructure.cpu.executor import CpuPool
    from ai_worker.infrastructure.embedding.raw import RawEmbedding


@runtime_checkable
class EmbeddingRuntime(Protocol):
    """Прогон модели над пачкой текстов."""

    async def encode(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[RawEmbedding]:
        """Считает сырые представления, ничего не интерпретируя."""
        ...


@dataclass(frozen=True, slots=True)
class PooledEncoderRuntime:
    """Прогон в отдельном процессе, где модель уже загружена."""

    pool: CpuPool
    model_dir: Path
    max_input_tokens: int

    async def prewarm(self, *, timeout_s: float) -> None:
        """Загружает модель в рабочий процесс до первого сообщения."""
        await self.pool.run_within(
            timeout_s,
            encoder.load,
            str(self.model_dir),
            self.max_input_tokens,
        )

    async def encode(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[RawEmbedding]:
        """Отдаёт пачку рабочему процессу и ждёт её не дольше отведённого."""
        return await self.pool.run_within(
            timeout_s,
            encoder.encode_batch,
            str(self.model_dir),
            self.max_input_tokens,
            tuple(texts),
        )
