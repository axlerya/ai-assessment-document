"""Технические границы value objects и namespace детерминированных идентификаторов."""

from __future__ import annotations

import uuid
from typing import Final

# Namespace UUIDv5 для page_id, span_id, chunk_id и event_id.
NS_DOCWORKER: Final[uuid.UUID] = uuid.UUID("6f1c0f8e-6a1e-5b2a-9f3c-2d4e5a6b7c81")

MIN_CORRELATION_ID_LENGTH: Final[int] = 8
MAX_CORRELATION_ID_LENGTH: Final[int] = 128

MIN_PAGE_NUMBER: Final[int] = 1
MAX_PAGES: Final[int] = 300
MAX_FILE_SIZE_BYTES: Final[int] = 104_857_600
SUPPORTED_MIME_TYPES: Final[frozenset[str]] = frozenset({"application/pdf"})

CONFIDENCE_PRECISION: Final[int] = 4
# Выше этого порога фрагмент нельзя звать неразборчивым.
MAX_ILLEGIBLE_CONFIDENCE: Final[float] = 0.75
MAX_PAGE_TEXT_LENGTH: Final[int] = 1_000_000

MIN_RENDER_DPI: Final[int] = 72
MAX_RENDER_DPI: Final[int] = 600

# Жёсткий потолок сущности, а не целевой размер чанка.
MAX_CHUNK_TOKENS: Final[int] = 1024
MAX_CHUNK_OVERLAP_CHARS: Final[int] = 400

MIN_BUCKET_NAME_LENGTH: Final[int] = 3
MAX_BUCKET_NAME_LENGTH: Final[int] = 63
MAX_OBJECT_KEY_LENGTH: Final[int] = 1024

MIN_VERSION_MAJOR: Final[int] = 1
MIN_VERSION_PART: Final[int] = 0
MAX_VERSION_PART: Final[int] = 999
