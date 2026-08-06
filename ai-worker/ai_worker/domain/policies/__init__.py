"""Доменные политики: чистые правила, возвращающие вердикт.

Список `ALL` существует не для удобства импорта, а для теста-стража: он
обходит модули политик и падает, если какая-нибудь из них начнёт строить
сущности вместо того, чтобы выносить вердикт.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_worker.domain.policies import (
    claim_grounding,
    draft_completeness,
    evidence_reliability,
)

if TYPE_CHECKING:
    from types import ModuleType

ALL: tuple[ModuleType, ...] = (
    evidence_reliability,
    claim_grounding,
    draft_completeness,
)
