"""Провайдер эмбеддингов: всё, что происходит вокруг прогона модели.

Таймаут вызова делится между пачками, а не выдаётся каждой целиком: иначе
заявленный предел превышался бы во столько раз, сколько пачек, и таймаут
обработки сообщения переставал бы что-либо ограничивать.
"""

from __future__ import annotations

import time
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_worker.application.errors import EmbeddingBackendUnavailable
from ai_worker.domain.value_objects.vectors import DenseVector, SparseVector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_worker.domain.embedding.policy import EmbeddingPolicy
    from ai_worker.infrastructure.embedding.raw import RawEmbedding
    from ai_worker.infrastructure.embedding.runtime import EmbeddingRuntime

Vectors = tuple[DenseVector, SparseVector]


@dataclass(frozen=True, slots=True)
class BgeM3EmbeddingProvider:
    """Строит плотное и разреженное представления одним проходом модели."""

    policy: EmbeddingPolicy
    runtime: EmbeddingRuntime
    batch_size: int

    async def embed_passages(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[Vectors]:
        """Считает представления фрагментов документа."""
        prepared = [f"{self.policy.passage_prefix}{text}" for text in texts]
        return await self._encode_all(prepared, timeout_s=timeout_s)

    async def embed_query(
        self,
        text: str,
        *,
        timeout_s: float,
    ) -> Vectors:
        """Считает представление пользовательского запроса."""
        prepared = f"{self.policy.query_prefix}{text}"
        vectors = await self._encode_all([prepared], timeout_s=timeout_s)
        return vectors[0]

    async def _encode_all(
        self,
        texts: Sequence[str],
        *,
        timeout_s: float,
    ) -> tuple[Vectors, ...]:
        if not texts:
            return ()
        deadline = time.monotonic() + timeout_s
        vectors: list[Vectors] = []
        for start in range(0, len(texts), self.batch_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingBackendUnavailable(
                    "таймаут вызова исчерпан до конца пачек",
                    context={"texts": len(texts), "done": len(vectors)},
                )
            batch = texts[start : start + self.batch_size]
            raw = await self._encode(batch, timeout_s=remaining)
            vectors.extend(self._to_vectors(item) for item in raw)
        return tuple(vectors)

    async def _encode(
        self,
        batch: Sequence[str],
        *,
        timeout_s: float,
    ) -> Sequence[RawEmbedding]:
        try:
            return await self.runtime.encode(batch, timeout_s=timeout_s)
        except TimeoutError as error:
            raise EmbeddingBackendUnavailable(
                "таймаут прогона модели",
                context={"batch": len(batch), "timeout_s": timeout_s},
            ) from error
        except BrokenProcessPool as error:
            # Рабочий процесс, убитый нехваткой памяти, — повод повторить, а не
            # отправить документ в разбор.
            raise EmbeddingBackendUnavailable(
                "рабочий процесс модели не пережил прогон",
                context={"batch": len(batch)},
            ) from error

    def _to_vectors(self, raw: RawEmbedding) -> Vectors:
        dense = DenseVector(raw.dense)
        if self.policy.normalize:
            dense = dense.normalized()
        return dense, SparseVector.pruned(raw.sparse, limit=self.policy.sparse_top_k)
