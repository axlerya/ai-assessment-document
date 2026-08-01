"""Порты распознавания.

Порт нейтрален: движок-специфичных параметров в нём нет. Правило — параметр
порта существует, только если его умеют все запланированные реализации;
`psm`, `oem` и прочая специфика одного движка делает абстракцию ложной.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from document_worker.application.dto.ocr import (
        OcrResult,
        PageImage,
        PreparedPage,
        PreprocessProfile,
    )


@runtime_checkable
class ImagePreprocessor(Protocol):
    """Готовит изображение страницы к распознаванию."""

    async def prepare(
        self,
        image: PageImage,
        *,
        profile: PreprocessProfile,
    ) -> PreparedPage:
        """Возвращает подготовленное изображение и его преобразование."""
        ...


@runtime_checkable
class OcrEngine(Protocol):
    """Распознаёт текст страницы."""

    async def recognize(
        self,
        page: PreparedPage,
        *,
        languages: Sequence[str],
        timeout_s: float,
        options: Mapping[str, str] | None = None,
    ) -> OcrResult:
        """Распознаёт страницу, не блокируя цикл событий.

        Raises:
            PageOcrTimeoutError: Страница не уложилась в отведённое время.
            OcrEngineError: Движок не смог выполнить вывод.
        """
        ...
