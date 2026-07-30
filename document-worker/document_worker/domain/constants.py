"""Технические границы value objects и namespace детерминированных идентификаторов."""

from __future__ import annotations

import uuid
from typing import Final

# Namespace UUIDv5 для page_id, span_id, chunk_id и event_id.
NS_DOCWORKER: Final[uuid.UUID] = uuid.UUID("6f1c0f8e-6a1e-5b2a-9f3c-2d4e5a6b7c81")

MIN_CORRELATION_ID_LENGTH: Final[int] = 8
MAX_CORRELATION_ID_LENGTH: Final[int] = 128

MIN_VERSION_MAJOR: Final[int] = 1
MIN_VERSION_PART: Final[int] = 0
MAX_VERSION_PART: Final[int] = 999
