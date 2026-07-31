"""Ошибки PDF-библиотек → прикладные.

Все они неисправимы: файл не станет читаемым от повтора. Ветвь одна, но классы
разные — по ним use case решает, что записать в отказ документа.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pikepdf

from document_worker.application.errors import (
    CorruptedDocumentError,
    EncryptedDocumentError,
)

if TYPE_CHECKING:
    from document_worker.application.errors import PermanentError


def translate_pdf_error(error: Exception, *, path: str) -> PermanentError:
    """Переводит ошибку разбора PDF в неисправимую прикладную."""
    context: dict[str, object] = {"path": path, "reason": type(error).__name__}
    if isinstance(error, pikepdf.PasswordError):
        translated: PermanentError = EncryptedDocumentError(
            "документ защищён паролем", context=context
        )
    else:
        translated = CorruptedDocumentError("документ не читается", context=context)
    translated.__cause__ = error
    return translated
