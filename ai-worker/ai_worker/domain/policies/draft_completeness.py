"""Полон ли черновик как документ.

Обязателен ровно один раздел — открытые вопросы. Это единственное место, где
сервис сообщает, чего в документах не нашлось, и его отсутствие читалось бы как
«вопросов нет». Остальные разделы необязательны: документ, в котором нет
денежных сумм, не должен получать выдуманный раздел про них.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_worker.domain.value_objects.enums import ClaimSection

REQUIRED_SECTIONS: frozenset[ClaimSection] = frozenset({ClaimSection.OPEN_QUESTIONS})


@dataclass(frozen=True, slots=True)
class CompletenessVerdict:
    """Собран ли черновик целиком и каких разделов не хватает."""

    complete: bool
    missing: tuple[ClaimSection, ...] = ()


@dataclass(frozen=True, slots=True)
class DraftCompletenessPolicy:
    """Требование к составу разделов сводки фактов."""

    def judge(self, *, sections: frozenset[ClaimSection]) -> CompletenessVerdict:
        """Выносит вердикт по набору заполненных разделов."""
        missing = tuple(
            section
            for section in ClaimSection
            if section in REQUIRED_SECTIONS - sections
        )
        return CompletenessVerdict(complete=not missing, missing=missing)
