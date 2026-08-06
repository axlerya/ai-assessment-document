"""Технические границы value objects и namespace детерминированных ключей."""

from __future__ import annotations

import uuid
from typing import Final

# Namespace UUIDv5 для embedding_id, index_id, draft_id, claim_id, citation_id
# и event_id. Свой, не совпадающий с namespace document-worker: общий namespace
# сделал бы ключи двух сервисов пересекающимися.
NS_AIWORKER: Final[uuid.UUID] = uuid.UUID("9b3f2d41-7c58-5e6a-8d1b-3f4a5c6d7e82")

# Ширина колонки `vector(1024)`: она задана миграцией и меняется только вместе
# с ней, поэтому проверяется в домене, а не в конфигурации.
DENSE_DIMENSIONS: Final[int] = 1024

# Словарь токенизатора XLM-RoBERTa, на котором построен bge-m3.
SPARSE_VOCABULARY_SIZE: Final[int] = 250_002
# Предел HNSW в pgvector: индекс по sparsevec отказывается строиться при
# большем числе ненулевых элементов.
SPARSE_TOP_K: Final[int] = 1_000

MIN_VERSION_MAJOR: Final[int] = 1
MIN_VERSION_PART: Final[int] = 0
MAX_VERSION_PART: Final[int] = 999
