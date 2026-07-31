"""Бюджет токенов чанкования и хэш параметров.

Полей со значениями по умолчанию у политики нет: пропуск поля обязан быть
`TypeError` конструктора, видимым mypy, а не тихим значением, разъезжающимся
с `.env.example`.

Хэш параметров сверяется с реестром на старте сервиса. Без этой сверки правка
бюджета через `.env` без инкремента версии сложила бы чанки, полученные разными
параметрами, в один namespace `chunking_version` — незаметно и необратимо:
дублей нет, ошибок нет, а границы фрагментов у половины документов другие.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_worker.domain.chunking.quality import (
    CHUNK_ILLEGIBLE_RATIO_THRESHOLD,
    MIN_LEGIBLE_CHARS,
    MIN_RETRIEVABLE_CONFIDENCE,
    MIN_RETRIEVABLE_TOKENS,
)
from document_worker.domain.chunking.structure_rules import (
    MAX_HEADING_CHARS,
    MAX_HEADING_ILLEGIBLE_RATIO,
    RULES_VERSION,
    TABLE_MIN_ROWS,
    UPPER_HEADING_PAGE_RATIO_CUTOFF,
)
from document_worker.domain.constants import MAX_CHUNK_TOKENS
from document_worker.domain.errors import InvalidChunkingPolicy
from document_worker.domain.value_objects.versioning import ChunkingVersion

if TYPE_CHECKING:
    from collections.abc import Mapping

CHUNKING_VERSION: Final[ChunkingVersion] = ChunkingVersion(1, 0, 0)


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    """Версия, кодировка токенизатора и бюджет токенов чанка."""

    version: ChunkingVersion
    encoding: str
    target_tokens: int
    max_tokens: int
    min_tokens: int
    overlap_tokens: int

    def __post_init__(self) -> None:
        """Сверяет бюджет с самим собой и с потолком сущности.

        Raises:
            InvalidChunkingPolicy: Бюджет противоречив либо выше потолка чанка.
        """
        if not self.encoding:
            self._reject("кодировка токенизатора не задана")
        if not 0 < self.min_tokens < self.target_tokens <= self.max_tokens:
            self._reject("бюджет не выстроен по возрастанию 0 < min < target <= max")
        # Без этой проверки InvariantViolation сущности прилетит уже в
        # терминальной части обработки первого плотного документа.
        if self.max_tokens > MAX_CHUNK_TOKENS:
            self._reject(f"предел чанка выше потолка сущности {MAX_CHUNK_TOKENS}")
        if not 0 <= self.overlap_tokens < self.min_tokens:
            self._reject("перекрытие не короче минимального чанка")

    def _reject(self, reason: str) -> None:
        raise InvalidChunkingPolicy(
            f"параметры чанкования непригодны: {reason}",
            context={"reason": reason},
        )

    def params_hash(self) -> str:
        """sha256 канонического представления всех параметров чанкования."""
        payload = json.dumps(
            {
                "chunking_version": str(self.version),
                "encoding": self.encoding,
                "rules_version": RULES_VERSION,
                "target_tokens": self.target_tokens,
                "max_tokens": self.max_tokens,
                "min_tokens": self.min_tokens,
                "overlap_tokens": self.overlap_tokens,
                "chunk_illegible_ratio_threshold": CHUNK_ILLEGIBLE_RATIO_THRESHOLD,
                "min_legible_chars": MIN_LEGIBLE_CHARS,
                "min_retrievable_confidence": MIN_RETRIEVABLE_CONFIDENCE,
                "min_retrievable_tokens": MIN_RETRIEVABLE_TOKENS,
                "max_heading_chars": MAX_HEADING_CHARS,
                "max_heading_illegible_ratio": MAX_HEADING_ILLEGIBLE_RATIO,
                "upper_heading_page_ratio_cutoff": UPPER_HEADING_PAGE_RATIO_CUTOFF,
                "table_min_rows": TABLE_MIN_ROWS,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ensure_registered(self) -> None:
        """Требует, чтобы параметрам соответствовала объявленная версия.

        Raises:
            InvalidChunkingPolicy: Версия неизвестна либо параметры под ней
                другие — чанки разных границ иначе попадут в один namespace.
        """
        recorded = KNOWN_PARAMS_HASHES.get(str(self.version))
        if recorded is None:
            raise InvalidChunkingPolicy(
                "версия чанкования не объявлена в реестре параметров",
                context={"version": str(self.version)},
            )
        actual = self.params_hash()
        if actual != recorded:
            raise InvalidChunkingPolicy(
                "параметры чанкования не соответствуют своей версии",
                context={
                    "version": str(self.version),
                    "expected": recorded,
                    "actual": actual,
                },
            )


DEFAULT_CHUNKING_POLICY: Final[ChunkingPolicy] = ChunkingPolicy(
    version=CHUNKING_VERSION,
    encoding="o200k_base",
    target_tokens=800,
    max_tokens=MAX_CHUNK_TOKENS,
    min_tokens=200,
    overlap_tokens=120,
)

# Хэш каждой выпущенной версии чанкования. Запись сюда — единственный способ
# изменить границы чанков, и она требует осознанного инкремента версии.
KNOWN_PARAMS_HASHES: Final[Mapping[str, str]] = {
    "1.0.0": "4213cdced5a38149ed16ccf439759ce01ef169f7ef04a80585813a7c0bfa6b4c",
}
