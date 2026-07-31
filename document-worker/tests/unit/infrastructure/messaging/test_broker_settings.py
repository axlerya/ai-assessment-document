"""Согласованность таймаутов доставки — проверка на старте, а не в бою."""

from __future__ import annotations

import pytest

from document_worker.application.errors import SchemaMisconfiguredError
from document_worker.infrastructure.messaging.broker import (
    PREFETCH_COUNT,
    ensure_delivery_settings_are_consistent,
)

pytestmark = pytest.mark.unit

CONSUMER_TIMEOUT_MS = 7_200_000
DOCUMENT_TIMEOUT_S = 3_600.0
CLAIM_LEASE_S = 5_400


def _check(**overrides: float) -> None:
    settings: dict[str, float] = {
        "prefetch_count": PREFETCH_COUNT,
        "consumer_timeout_ms": CONSUMER_TIMEOUT_MS,
        "document_timeout_s": DOCUMENT_TIMEOUT_S,
        "claim_lease_s": CLAIM_LEASE_S,
    }
    settings.update(overrides)
    ensure_delivery_settings_are_consistent(**settings)  # type: ignore[arg-type]


def test_calibrated_settings_are_accepted() -> None:
    _check()


def test_prefetch_above_one_with_long_document_timeout_is_rejected() -> None:
    # Сообщения со второго по N-е протухнут в доставленных раньше, чем воркер
    # до них дойдёт.
    with pytest.raises(SchemaMisconfiguredError):
        _check(prefetch_count=4)


def test_prefetch_above_one_with_short_document_timeout_is_accepted() -> None:
    _check(prefetch_count=4, document_timeout_s=600.0)


def test_consumer_timeout_not_greater_than_document_timeout_is_rejected() -> None:
    # Таймер идёт с момента доставки: равные значения дают вечный цикл
    # возвратов на самом длинном документе.
    with pytest.raises(SchemaMisconfiguredError):
        _check(consumer_timeout_ms=int(DOCUMENT_TIMEOUT_S * 1000))


def test_claim_lease_shorter_than_document_timeout_is_rejected() -> None:
    # Лиз, протухающий в середине обработки, делает каждую повторную доставку
    # возобновлением параллельно живому воркеру.
    with pytest.raises(SchemaMisconfiguredError):
        _check(claim_lease_s=1_800)
